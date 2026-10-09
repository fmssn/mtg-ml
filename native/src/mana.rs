//! Mana costs, remaining costs and payment feasibility (mirror of mana.py).
//! Mana types are bytes: b'W', b'U', b'B', b'R', b'G', b'C'.

pub const MANA_TYPES: [u8; 6] = [b'W', b'U', b'B', b'R', b'G', b'C'];

/// Bit for a mana type inside a unit's alternative set.
pub fn bit(c: u8) -> u8 {
    match c {
        b'W' => 1,
        b'U' => 2,
        b'B' => 4,
        b'R' => 8,
        b'G' => 16,
        b'C' => 32,
        _ => panic!("bad mana type {}", c as char),
    }
}

/// `{R/P}`: a phyrexian symbol of a colour.
fn is_phyrexian(sym: &str) -> bool {
    let b = sym.as_bytes();
    b.len() == 3 && &sym[1..] == "/P" && MANA_TYPES[..5].contains(&b[0])
}

#[derive(Clone, Debug, PartialEq, Eq, Default)]
pub struct ManaCost {
    pub generic: i32,
    /// Sorted by mana-type character (Python: `tuple(sorted(colored.items()))`).
    pub colored: Vec<(u8, i32)>,
    pub x: i32,
}

impl ManaCost {
    pub fn generic(n: i32) -> Self {
        ManaCost { generic: n, colored: vec![], x: 0 }
    }

    pub fn parse(text: Option<&str>) -> Result<Self, String> {
        let text = match text {
            None | Some("") => return Ok(ManaCost::default()),
            Some(t) => t,
        };
        let mut generic = 0;
        let mut x = 0;
        let mut colored: Vec<(u8, i32)> = vec![];
        let mut rest = text;
        while let Some(start) = rest.find('{') {
            let end = rest[start..].find('}').ok_or_else(|| format!("unterminated symbol in {text:?}"))? + start;
            let sym = &rest[start + 1..end];
            if !sym.is_empty() && sym.bytes().all(|b| b.is_ascii_digit()) {
                generic += sym.parse::<i32>().unwrap();
            } else if sym == "X" {
                x += 1;
            } else if (sym.len() == 1 && MANA_TYPES.contains(&sym.as_bytes()[0])) || is_phyrexian(sym) {
                // Phyrexian mana counts as its colour here; paying 2 life is the cast mode "phyrexian".
                let c = sym.as_bytes()[0];
                match colored.iter_mut().find(|(k, _)| *k == c) {
                    Some(e) => e.1 += 1,
                    None => colored.push((c, 1)),
                }
            } else {
                return Err(format!("unsupported mana symbol {{{sym}}} in {text:?}"));
            }
            rest = &rest[end + 1..];
        }
        colored.sort();
        Ok(ManaCost { generic, colored, x })
    }

    /// The phyrexian symbols of a cost text ({R/P} -> {R}), as a cost.
    pub fn phyrexian(text: Option<&str>) -> ManaCost {
        let mut colored: Vec<(u8, i32)> = vec![];
        let mut rest = text.unwrap_or("");
        while let Some(start) = rest.find('{') {
            let Some(len) = rest[start..].find('}') else { break };
            let sym = &rest[start + 1..start + len];
            if is_phyrexian(sym) {
                let c = sym.as_bytes()[0];
                match colored.iter_mut().find(|(k, _)| *k == c) {
                    Some(e) => e.1 += 1,
                    None => colored.push((c, 1)),
                }
            }
            rest = &rest[start + len + 1..];
        }
        colored.sort();
        ManaCost { generic: 0, colored, x: 0 }
    }

    /// This cost without `other`'s coloured symbols.
    pub fn minus_colored(&self, other: &ManaCost) -> ManaCost {
        let mut colored = self.colored.clone();
        for (k, n) in &other.colored {
            if let Some(e) = colored.iter_mut().find(|(c, _)| c == k) {
                e.1 -= n;
            }
        }
        colored.retain(|(_, n)| *n > 0);
        ManaCost { generic: self.generic, colored, x: self.x }
    }

    pub fn mana_value(&self) -> i32 {
        self.generic + self.colored.iter().map(|(_, n)| n).sum::<i32>()
    }

    pub fn with_x(&self, x_value: i32) -> ManaCost {
        ManaCost { generic: self.generic + x_value * self.x, colored: self.colored.clone(), x: 0 }
    }

    pub fn reduced(&self, amount: i32) -> ManaCost {
        ManaCost { generic: (self.generic - amount).max(0), colored: self.colored.clone(), x: self.x }
    }

    pub fn is_zero(&self) -> bool {
        self.generic == 0 && self.colored.is_empty() && self.x == 0
    }

    pub fn to_string(&self) -> String {
        let mut s = String::new();
        for _ in 0..self.x {
            s.push_str("{X}");
        }
        if self.generic != 0 || (self.colored.is_empty() && self.x == 0) {
            s.push_str(&format!("{{{}}}", self.generic));
        }
        for t in MANA_TYPES {
            if let Some((_, n)) = self.colored.iter().find(|(k, _)| *k == t) {
                for _ in 0..*n {
                    s.push('{');
                    s.push(t as char);
                    s.push('}');
                }
            }
        }
        s
    }
}

/// Mutable remainder of a cost being paid. `colored` keeps Python dict order
/// (sorted at creation, entries deleted when they reach zero).
#[derive(Clone, Debug)]
pub struct Remaining {
    pub generic: i32,
    pub colored: Vec<(u8, i32)>,
}

impl Remaining {
    pub fn of(cost: &ManaCost) -> Remaining {
        assert!(cost.x == 0, "X must be fixed before payment");
        Remaining { generic: cost.generic, colored: cost.colored.clone() }
    }

    pub fn is_paid(&self) -> bool {
        self.generic == 0 && self.colored.iter().all(|(_, n)| *n == 0)
    }

    pub fn colored_get(&self, c: u8) -> i32 {
        self.colored.iter().find(|(k, _)| *k == c).map(|e| e.1).unwrap_or(0)
    }

    pub fn apply(&mut self, mana: u8) -> bool {
        if let Some(pos) = self.colored.iter().position(|(k, n)| *k == mana && *n > 0) {
            self.colored[pos].1 -= 1;
            if self.colored[pos].1 == 0 {
                self.colored.remove(pos);
            }
            return true;
        }
        if self.generic > 0 {
            self.generic -= 1;
            return true;
        }
        false
    }

    pub fn useful(&self, mana: u8) -> bool {
        self.colored_get(mana) > 0 || self.generic > 0
    }

    pub fn to_string(&self) -> String {
        let mut colored = self.colored.clone();
        colored.sort();
        ManaCost { generic: self.generic, colored, x: 0 }.to_string()
    }
}

/// Can `rem` be paid with `units` (each a bitmask of the mana types that unit
/// can be)? Kuhn's augmenting-path matching of coloured symbols to units.
pub fn can_pay(rem: &Remaining, units: &[u8]) -> bool {
    can_pay_wild(rem, units, 0)
}

/// mana.py `can_pay(..., wild)`: up to `wild` coloured (WUBRG) symbols may
/// go unmatched (mana filters pay them as generic); {C} symbols come first
/// and must all be matched.
pub fn can_pay_wild(rem: &Remaining, units: &[u8], wild: i32) -> bool {
    let mut symbols: Vec<u8> = Vec::with_capacity(8);
    for (c, n) in &rem.colored {
        if *c == b'C' {
            for _ in 0..*n {
                symbols.push(bit(*c));
            }
        }
    }
    let n_c = symbols.len();
    for (c, n) in &rem.colored {
        if *c != b'C' {
            for _ in 0..*n {
                symbols.push(bit(*c));
            }
        }
    }
    if (units.len() as i32) < symbols.len() as i32 + rem.generic {
        return false;
    }
    if symbols.is_empty() {
        return true;
    }
    let mut match_unit: Vec<i32> = vec![-1; units.len()];
    fn augment(si: usize, symbols: &[u8], units: &[u8], seen: &mut [bool], match_unit: &mut [i32]) -> bool {
        for ui in 0..units.len() {
            if units[ui] & symbols[si] != 0 && !seen[ui] {
                seen[ui] = true;
                if match_unit[ui] == -1 || augment(match_unit[ui] as usize, symbols, units, seen, match_unit) {
                    match_unit[ui] = si as i32;
                    return true;
                }
            }
        }
        false
    }
    let mut seen = vec![false; units.len()];
    let mut unmatched = 0;
    for si in 0..symbols.len() {
        seen.iter_mut().for_each(|s| *s = false);
        if !augment(si, &symbols, units, &mut seen, &mut match_unit) {
            if si < n_c {
                return false;
            }
            unmatched += 1;
            if unmatched > wild {
                return false;
            }
        }
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_and_format() {
        let c = ManaCost::parse(Some("{X}{G}{G}")).unwrap();
        assert_eq!(c.to_string(), "{X}{G}{G}");
        assert_eq!(ManaCost::parse(Some("{5}{U}{U}")).unwrap().to_string(), "{5}{U}{U}");
        assert_eq!(ManaCost::default().to_string(), "{0}");
        assert_eq!(ManaCost::parse(Some("{B}{R}{G}")).unwrap().colored, vec![(b'B', 1), (b'G', 1), (b'R', 1)]);
        assert_eq!(ManaCost::parse(Some("{B}{R}{G}")).unwrap().to_string(), "{B}{R}{G}");
    }

    #[test]
    fn matching() {
        let r = Remaining::of(&ManaCost::parse(Some("{1}{B}{R}")).unwrap());
        assert!(can_pay(&r, &[bit(b'B') | bit(b'R'), bit(b'R'), bit(b'C')]));
        assert!(!can_pay(&r, &[bit(b'B') | bit(b'R'), bit(b'C'), bit(b'C')]));
    }
}
