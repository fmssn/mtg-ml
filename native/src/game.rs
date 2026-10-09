//! Driver: owns the state and the engine coroutine (Python's generator).

use corosensei::stack::DefaultStack;
use corosensei::{Coroutine, CoroutineResult};
use std::cell::RefCell;

use crate::engine::Eng;
use crate::state::{Args, State, Stop, ABORT, R};

const STACK_SIZE: usize = 256 * 1024;
/// Finished games hand their coroutine stack to the next game on the same
/// thread: allocating one (mmap + guard page) is a large part of a `copy`.
const POOLED_STACKS: usize = 64;

thread_local! {
    static STACKS: RefCell<Vec<DefaultStack>> = const { RefCell::new(Vec::new()) };
}

pub enum StepError {
    Over,
    Index(usize, usize),
    Rules(String),
}

pub struct Game {
    st: *mut State,
    co: Option<Coroutine<usize, (), R<()>, DefaultStack>>,
    started: bool,
    /// Set when the engine raised a RulesError: the game cannot continue.
    pub broken: Option<String>,
}

impl Game {
    /// Python `Game.__init__` up to (excluding) `setup(self)`; call `start()`
    /// after the setup has run.
    pub fn new(args: Args) -> Result<Game, String> {
        let st = Box::new(State::new(args)?);
        Ok(Game { st: Box::into_raw(st), co: None, started: false, broken: None })
    }

    #[inline]
    pub fn state(&self) -> &State {
        unsafe { &*self.st }
    }

    /// Mutable access. Only sound while the engine is not running, which is
    /// always the case from outside (the coroutine runs inside `step`).
    #[inline]
    pub fn state_mut(&mut self) -> &mut State {
        unsafe { &mut *self.st }
    }

    pub fn started(&self) -> bool {
        self.started
    }

    pub fn start(&mut self) -> Result<(), String> {
        // Edits made before the start (a scenario setup) are part of the
        // construction, which a replay repeats.
        self.state_mut().edited = false;
        self.start_with(None)
    }

    fn start_with(&mut self, resume: Option<(&'static str, bool)>) -> Result<(), String> {
        if self.started {
            return Err("game already started".into());
        }
        self.started = true;
        let sptr = self.st;
        let start_step = self.state().args.start_step.clone();
        let stack = match STACKS.with(|p| p.borrow_mut().pop()) {
            Some(s) => s,
            None => DefaultStack::new(STACK_SIZE).map_err(|e| format!("coroutine stack: {e}"))?,
        };
        self.co = Some(Coroutine::with_stack(stack, move |y, _input: usize| {
            let mut e = Eng { s: sptr, y: y as *const _ };
            e.main(&start_step, resume)
        }));
        self.advance(0).map_err(|e| match e {
            StepError::Rules(m) => m,
            _ => unreachable!(),
        })
    }

    /// `Game.copy`: an independent game in exactly this state (RNG included)
    /// that continues identically. The suspended coroutine cannot be cloned,
    /// so the copy restarts the engine from the latest step-start snapshot
    /// and replays the actions taken since. `Ok(None)` when there is no
    /// usable snapshot: the caller replays the whole history instead. Either
    /// way this game takes snapshots from now on; the copy does not until it
    /// is copied itself (it shares this one's until its next step begins).
    pub fn copy(&mut self) -> Result<Option<Game>, StepError> {
        if let Some(m) = &self.broken {
            return Err(StepError::Rules(format!("engine stopped after an error: {m}")));
        }
        self.state_mut().snapshots = true;
        let snap = match (&self.state().snap, self.started) {
            (Some(s), true) => s.clone(),
            _ => return Ok(None),
        };
        let mut st = snap.state.clone();
        st.snap = Some(snap.clone());
        st.snapshots = false;
        st.edited = false;
        let mut g = Game { st: Box::into_raw(Box::new(st)), co: None, started: false, broken: None };
        g.start_with(Some((snap.step, snap.skip_draw))).map_err(StepError::Rules)?;
        for &a in &self.state().actions[snap.n_actions..] {
            g.step(a as usize)?;
        }
        Ok(Some(g))
    }

    fn advance(&mut self, input: usize) -> Result<(), StepError> {
        let res = self.co.as_mut().expect("not started").resume(input);
        match res {
            CoroutineResult::Yield(()) => Ok(()),
            CoroutineResult::Return(Ok(())) => unreachable!("engine main loop returned"),
            CoroutineResult::Return(Err(Stop::GameOver { winner, reason })) => {
                let st = self.state_mut();
                st.over = true;
                st.winner = winner;
                st.end_reason = reason;
                st.decision = None;
                if st.logging {
                    let w = winner.map_or("None".to_string(), |w| w.to_string());
                    st.log.push(format!("GAME OVER: winner={w} ({reason})"));
                }
                self.release_co();
                Ok(())
            }
            CoroutineResult::Return(Err(Stop::Abort)) => unreachable!("aborted outside release_co"),
            CoroutineResult::Return(Err(Stop::Rules(m))) => {
                self.state_mut().decision = None;
                self.broken = Some(m.clone());
                self.release_co();
                Err(StepError::Rules(m))
            }
        }
    }

    /// Drop the engine coroutine (unwinding it if suspended) and keep its
    /// stack for the next game on this thread.
    fn release_co(&mut self) {
        if let Some(mut co) = self.co.take() {
            if co.started() && !co.done() {
                // Suspended in `ask`: let it return Stop::Abort up to `main`.
                if let CoroutineResult::Yield(()) = co.resume(ABORT) {
                    co.force_unwind();
                }
            }
            let stack = co.into_stack();
            STACKS.with(|p| {
                let mut p = p.borrow_mut();
                if p.len() < POOLED_STACKS {
                    p.push(stack);
                }
            });
        }
    }

    pub fn step(&mut self, index: usize) -> Result<(), StepError> {
        if let Some(m) = &self.broken {
            return Err(StepError::Rules(format!("engine stopped after an error: {m}")));
        }
        let st = self.state_mut();
        let d = match (&st.decision, st.over) {
            (Some(d), false) => d,
            _ => return Err(StepError::Over),
        };
        if index >= d.options.len() {
            return Err(StepError::Index(index, d.options.len()));
        }
        st.actions.push(index as u32);
        if st.logging {
            let m = format!("  p{} {}: {}", d.player, d.kind.name(), d.options[index].label);
            st.log.push(m);
        }
        self.advance(index)
    }
}

impl Drop for Game {
    fn drop(&mut self) {
        // Unwind a suspended engine before freeing the state it points to.
        self.release_co();
        unsafe { drop(Box::from_raw(self.st)) };
    }
}
