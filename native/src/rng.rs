//! CPython's `random.Random`, bit for bit: MT19937 seeded with
//! `init_by_array` from the absolute value of an integer seed, `getrandbits`,
//! `_randbelow` (rejection sampling on `getrandbits(n.bit_length())`) and
//! `shuffle`. The engine only ever calls `randrange(2)` and `shuffle`.

const N: usize = 624;
const M: usize = 397;

#[derive(Clone)]
pub struct PyRandom {
    mt: [u32; N],
    index: usize,
}

impl PyRandom {
    /// `random.Random(seed)` for an integer seed (CPython hashes nothing for
    /// ints: the key is |seed| split into little-endian 32-bit words).
    pub fn new(seed: i128) -> Self {
        let mut n = seed.unsigned_abs();
        let mut key = Vec::new();
        while n > 0 {
            key.push((n & 0xffff_ffff) as u32);
            n >>= 32;
        }
        if key.is_empty() {
            key.push(0);
        }
        let mut r = PyRandom { mt: [0; N], index: N };
        r.init_by_array(&key);
        r
    }

    fn init_genrand(&mut self, s: u32) {
        self.mt[0] = s;
        for i in 1..N {
            let prev = self.mt[i - 1];
            self.mt[i] = 1812433253u32.wrapping_mul(prev ^ (prev >> 30)).wrapping_add(i as u32);
        }
        self.index = N;
    }

    fn init_by_array(&mut self, key: &[u32]) {
        self.init_genrand(19650218);
        let mut i = 1usize;
        let mut j = 0usize;
        let mut k = N.max(key.len());
        while k > 0 {
            let prev = self.mt[i - 1];
            self.mt[i] = (self.mt[i] ^ (prev ^ (prev >> 30)).wrapping_mul(1664525)).wrapping_add(key[j]).wrapping_add(j as u32);
            i += 1;
            j += 1;
            if i >= N {
                self.mt[0] = self.mt[N - 1];
                i = 1;
            }
            if j >= key.len() {
                j = 0;
            }
            k -= 1;
        }
        k = N - 1;
        while k > 0 {
            let prev = self.mt[i - 1];
            self.mt[i] = (self.mt[i] ^ (prev ^ (prev >> 30)).wrapping_mul(1566083941)).wrapping_sub(i as u32);
            i += 1;
            if i >= N {
                self.mt[0] = self.mt[N - 1];
                i = 1;
            }
            k -= 1;
        }
        self.mt[0] = 0x8000_0000;
    }

    pub fn genrand_u32(&mut self) -> u32 {
        const UPPER: u32 = 0x8000_0000;
        const LOWER: u32 = 0x7fff_ffff;
        const MATRIX_A: u32 = 0x9908_b0df;
        if self.index >= N {
            for kk in 0..N {
                let y = (self.mt[kk] & UPPER) | (self.mt[(kk + 1) % N] & LOWER);
                let mag = if y & 1 != 0 { MATRIX_A } else { 0 };
                self.mt[kk] = self.mt[(kk + M) % N] ^ (y >> 1) ^ mag;
            }
            self.index = 0;
        }
        let mut y = self.mt[self.index];
        self.index += 1;
        y ^= y >> 11;
        y ^= (y << 7) & 0x9d2c_5680;
        y ^= (y << 15) & 0xefc6_0000;
        y ^= y >> 18;
        y
    }

    /// `getrandbits(k)` for 0 <= k <= 32.
    pub fn getrandbits(&mut self, k: u32) -> u32 {
        if k == 0 {
            return 0;
        }
        debug_assert!(k <= 32);
        self.genrand_u32() >> (32 - k)
    }

    /// `_randbelow(n)` (n < 2**32).
    pub fn randbelow(&mut self, n: u32) -> u32 {
        if n == 0 {
            return 0;
        }
        let k = 32 - n.leading_zeros();
        let mut r = self.getrandbits(k);
        while r >= n {
            r = self.getrandbits(k);
        }
        r
    }

    pub fn shuffle<T>(&mut self, x: &mut [T]) {
        for i in (1..x.len()).rev() {
            let j = self.randbelow(i as u32 + 1) as usize;
            x.swap(i, j);
        }
    }

    pub fn set_state(&mut self, words: &[u32]) -> Result<(), String> {
        if words.len() != N + 1 || words[N] as usize > N {
            return Err("invalid MT19937 state".into());
        }
        self.mt.copy_from_slice(&words[..N]);
        self.index = words[N] as usize;
        Ok(())
    }

    /// `getstate()[1]`: the 624 state words followed by the index.
    pub fn state(&self) -> Vec<u32> {
        let mut v = self.mt.to_vec();
        v.push(self.index as u32);
        v
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_cpython() {
        // random.Random(0).getrandbits(32) x3 and random.Random(12345).randrange(2) x8
        let mut r = PyRandom::new(0);
        assert_eq!([r.genrand_u32(), r.genrand_u32(), r.genrand_u32()], [3626764237, 1654615998, 3255389356]);
        let mut r = PyRandom::new(1 << 40);
        let mut v: Vec<u32> = (0..10).collect();
        r.shuffle(&mut v);
        assert_eq!(v, [7, 8, 2, 4, 6, 0, 3, 9, 5, 1]);
        let mut r = PyRandom::new(-7);
        let mut v: Vec<u32> = (0..60).collect();
        r.shuffle(&mut v);
        assert_eq!(v[..10], [12, 45, 10, 40, 50, 53, 30, 16, 0, 19]);
        assert_eq!(r.randbelow(2), 1);
    }
}
