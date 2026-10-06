//! Driver: owns the state and the engine coroutine (Python's generator).

use corosensei::stack::DefaultStack;
use corosensei::{Coroutine, CoroutineResult};

use crate::engine::Eng;
use crate::state::{Args, State, Stop, R};

const STACK_SIZE: usize = 256 * 1024;

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
        if self.started {
            return Err("game already started".into());
        }
        self.started = true;
        let sptr = self.st;
        let start_step = self.state().args.start_step.clone();
        let stack = DefaultStack::new(STACK_SIZE).map_err(|e| format!("coroutine stack: {e}"))?;
        self.co = Some(Coroutine::with_stack(stack, move |y, _input: usize| {
            let mut e = Eng { s: sptr, y: y as *const _ };
            e.main(&start_step)
        }));
        self.advance(0).map_err(|e| match e {
            StepError::Rules(m) => m,
            _ => unreachable!(),
        })
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
                self.co = None;
                Ok(())
            }
            CoroutineResult::Return(Err(Stop::Rules(m))) => {
                self.state_mut().decision = None;
                self.broken = Some(m.clone());
                self.co = None;
                Err(StepError::Rules(m))
            }
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
        self.co = None;
        unsafe { drop(Box::from_raw(self.st)) };
    }
}
