from mtg_ml.engine.mana import ManaCost, RemainingCost, can_pay


def test_parse_and_str():
    c = ManaCost.parse("{5}{U}{U}")
    assert c.generic == 5 and c.colored_dict() == {"U": 2} and c.mana_value == 7
    assert str(c) == "{5}{U}{U}"
    x = ManaCost.parse("{X}{G}{G}")
    assert x.x == 1 and str(x.with_x(3)) == "{3}{G}{G}"
    assert str(ManaCost.parse("{6}{U}").reduced(10)) == "{U}"


def test_can_pay_matching():
    rem = RemainingCost.of(ManaCost.parse("{1}{B}{R}"))
    assert can_pay(rem, [("B", "R"), ("R", "G"), ("C",)])
    assert not can_pay(rem, [("B",), ("B",), ("C",)])  # no red source
    assert not can_pay(rem, [("B", "R"), ("R", "G")])  # too few units
    # one dual must cover B and another R: matching, not greedy
    assert can_pay(RemainingCost.of(ManaCost.parse("{B}{R}")), [("B", "R"), ("B",)])


def test_remaining_prefers_colored():
    rem = RemainingCost.of(ManaCost.parse("{1}{U}"))
    assert rem.apply("U") and rem.colored == {} and rem.generic == 1
    assert rem.apply("B") and rem.is_paid()
    assert not rem.apply("U")
