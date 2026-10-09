//! The card pool, loaded from the same `cards.toml` the Python engine uses.
//!
//! The Python wrapper passes the spec text in at import time
//! (`mtg_ml_native.load_cards`), so a card that only uses existing ops needs
//! no rebuild. Without that call (Rust unit tests) the copy compiled into the
//! crate is used.

use std::collections::HashMap;
use std::sync::OnceLock;

use toml::{Table, Value};

use crate::mana::ManaCost;

pub type DefId = u16;

pub const BUILTIN_SPEC: &str = include_str!("../../mtg_ml/engine/cards.toml");

// Card types, bit order = alphabetical so iteration yields sorted names.
pub const TYPE_NAMES: [&str; 9] = ["Artifact", "Battle", "Creature", "Enchantment", "Instant", "Kindred", "Land", "Planeswalker", "Sorcery"];
pub const T_ARTIFACT: u16 = 1 << 0;
pub const T_BATTLE: u16 = 1 << 1;
pub const T_CREATURE: u16 = 1 << 2;
pub const T_ENCHANTMENT: u16 = 1 << 3;
pub const T_INSTANT: u16 = 1 << 4;
pub const T_LAND: u16 = 1 << 6;
pub const T_PLANESWALKER: u16 = 1 << 7;
pub const T_SORCERY: u16 = 1 << 8;

pub fn type_bit(name: &str) -> Result<u16, String> {
    TYPE_NAMES.iter().position(|t| *t == name).map(|i| 1u16 << i).ok_or_else(|| format!("unknown card type {name:?}"))
}

pub fn type_names(mask: u16) -> Vec<&'static str> {
    (0..TYPE_NAMES.len()).filter(|i| mask & (1 << i) != 0).map(|i| TYPE_NAMES[i]).collect()
}

/// Colour bits (W U B R G), as in a card's colours.
pub fn color_bit(c: u8) -> u8 {
    match c {
        b'W' => 1,
        b'U' => 2,
        b'B' => 4,
        b'R' => 8,
        b'G' => 16,
        _ => 0,
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TK {
    Creature,
    NonlegendaryCreature,
    NonartifactCreature,
    CreatureYouControl,
    Land,
    NonlandPermanent,
    Artifact,
    BluePermanent,
    RedPermanent,
    Spell,
    BlueSpell,
    RedSpell,
    InstantSpell,
    ArtifactSpell,
    Player,
    Opponent,
    Any,
    CreatureYouDontControl,
    Permanent,
    NoncreatureArtifact,
    PlayerWithCreature,
    /// A creature controlled by the player chosen as the previous target.
    CreatureOfTargetPlayer,
    ArtifactOrEnchantmentSpell,
    /// A creature not already chosen as a target of the same spell.
    AnotherCreature,
    ArtifactOrEnchantmentYouDontControl,
    SorcerySpell,
    ArtifactOrEnchantment,
    NoncreatureSpell,
}

const TK_NAMES: [(&str, TK); 28] = [
    ("creature", TK::Creature),
    ("nonlegendary_creature", TK::NonlegendaryCreature),
    ("nonartifact_creature", TK::NonartifactCreature),
    ("creature_you_control", TK::CreatureYouControl),
    ("land", TK::Land),
    ("nonland_permanent", TK::NonlandPermanent),
    ("artifact", TK::Artifact),
    ("blue_permanent", TK::BluePermanent),
    ("red_permanent", TK::RedPermanent),
    ("spell", TK::Spell),
    ("blue_spell", TK::BlueSpell),
    ("red_spell", TK::RedSpell),
    ("instant_spell", TK::InstantSpell),
    ("artifact_spell", TK::ArtifactSpell),
    ("player", TK::Player),
    ("opponent", TK::Opponent),
    ("any", TK::Any),
    ("creature_you_dont_control", TK::CreatureYouDontControl),
    ("permanent", TK::Permanent),
    ("noncreature_artifact", TK::NoncreatureArtifact),
    ("player_with_creature", TK::PlayerWithCreature),
    ("creature_of_target_player", TK::CreatureOfTargetPlayer),
    ("artifact_or_enchantment_spell", TK::ArtifactOrEnchantmentSpell),
    ("another_creature", TK::AnotherCreature),
    ("artifact_or_enchantment_you_dont_control", TK::ArtifactOrEnchantmentYouDontControl),
    ("sorcery_spell", TK::SorcerySpell),
    ("artifact_or_enchantment", TK::ArtifactOrEnchantment),
    ("noncreature_spell", TK::NoncreatureSpell),
];

impl TK {
    pub fn parse(s: &str) -> Result<TK, String> {
        TK_NAMES.iter().find(|(n, _)| *n == s).map(|(_, k)| *k).ok_or_else(|| format!("unknown target kind {s:?}"))
    }
    pub fn name(self) -> &'static str {
        TK_NAMES.iter().find(|(_, k)| *k == self).unwrap().0
    }
    pub fn is_spell(self) -> bool {
        self.name().ends_with("spell")
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SacFilter {
    Artifact,
    ArtifactOrCreature,
    Mountain,
    /// Bargain: an artifact, enchantment or token.
    ArtifactEnchantmentOrToken,
    Land,
    /// Not a sacrifice: a creature must stay behind while paying (Monstrous Emergence).
    KeepCreature,
}

impl SacFilter {
    pub fn parse(s: &str) -> Result<SacFilter, String> {
        match s {
            "artifact" => Ok(SacFilter::Artifact),
            "artifact_or_creature" => Ok(SacFilter::ArtifactOrCreature),
            "mountain" => Ok(SacFilter::Mountain),
            "artifact_enchantment_or_token" => Ok(SacFilter::ArtifactEnchantmentOrToken),
            "land" => Ok(SacFilter::Land),
            "keep_creature" => Ok(SacFilter::KeepCreature),
            _ => Err(format!("unknown sacrifice filter {s:?}")),
        }
    }
    pub fn name(self) -> &'static str {
        match self {
            SacFilter::Artifact => "artifact",
            SacFilter::ArtifactOrCreature => "artifact_or_creature",
            SacFilter::Mountain => "mountain",
            SacFilter::ArtifactEnchantmentOrToken => "artifact_enchantment_or_token",
            SacFilter::Land => "land",
            SacFilter::KeepCreature => "keep_creature",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CostRed {
    InstantsAndSorceriesInGraveyard,
    ArtifactsYouControl,
    CardsDrawnThisTurn,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Event {
    Etb,
    ToGraveyardFromBattlefield,
    Cast,
    YouSacrificeAnother,
    YourUpkeep,
    BecomesTarget,
    /// Another spell its controller casts (from the battlefield).
    YouCast,
    /// Its owner draws their third card in a turn (from the graveyard).
    ThirdDraw,
    /// It leaves the battlefield (to any zone).
    LeavesBattlefield,
    /// Engine-internal: the madness trigger.
    Discarded,
    /// A dungeon room: the venture marker moved into it.
    Room,
    /// Engine-internal: the initiative's inherent triggers.
    Initiative,
}

/// `you_cast` trigger condition `{ spell = ... }`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CastFilter {
    Noncreature,
    InstantOrSorcery,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Custom {
    DelverReveal,
    Brainstorm,
    Ponder,
    DeemInferior,
    OpponentDiscardsElseDraw,
    Wildfire,
    Duress,
    HighwayRobbery,
    RelicExileOne,
}

/// Whose life an op changes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Who {
    You,
    Opponent,
    TargetPlayer,
    TargetController,
}

#[derive(Clone, Debug)]
pub struct SearchFilter {
    pub supertype: Option<String>,
    pub types: u16,
    pub subtypes_any: Vec<String>,
    pub colorless: bool,
}

#[derive(Clone, Debug)]
pub enum Op {
    /// each_controlling: each player who controls a permanent with this name draws instead.
    Draw { n: i32, n_cast_from_graveyard: Option<i32>, each_controlling: Option<String>, target_player: bool },
    Mill { target_player: bool, n: i32 },
    /// if_color: colour bit the target spell must have (0 = any).
    CounterTarget { if_color: u8 },
    CounterTargetUnlessPaid { cost: ManaCost },
    DestroyTarget { if_color: u8, mv_is_x: bool },
    BounceTarget,
    TapTarget { skip_untap: i32 },
    GrantTarget { keywords: u32 },
    /// attach_source: then attach the source Equipment to the token (job select).
    CreateToken { token: DefId, n: i32, attach_source: bool },
    /// Equip: attach the source Equipment to the target creature.
    AttachSourceToTarget,
    /// per_storm: n for each spell cast before this one this turn (Weather the Storm's storm trigger);
    /// sacrificed_mv: instead of n, the mana value of the permanent sacrificed to cast it;
    /// source_power: n_from = "source_power" (life equal to the source's power).
    GainLife { n: i32, per_storm: bool, sacrificed_mv: bool, source_power: bool },
    LoseLife { who: Who, n: i32 },
    CounterOnSource,
    DamageTarget { n: i32, index: usize, n_landfall: Option<i32>, n_metalcraft: Option<i32> },
    /// To the controller of the targeted permanent.
    DamageTargetController { n: i32 },
    /// damage_target_from: X damage (from = "x") or the power of the creature
    /// chosen as the spell was cast (from = "chosen_power", Monstrous Emergence).
    DamageTargetFrom { index: usize, chosen_power: bool },
    DamageEachOpponent { n: i32, if_discarded_nonland: bool },
    /// x: the amount is the item's X (n unused); opponent_only: whose = "opponent".
    /// except_subtype: a creature type that is spared, changelings included (Fiery Cannonade).
    DamageEachCreature { n: i32, x: bool, without: u32, opponent_only: bool, except_subtype: Option<String> },
    Discard { n: i32 },
    ReturnToBattlefield { tapped: bool },
    ExileGraveyard,
    ExileAllGraveyards,
    ExileTarget,
    /// Up to n cards from any graveyards, chosen as it resolves.
    ExileFromGraveyards { n: i32 },
    SearchLibrary { filter: SearchFilter, to_battlefield: bool, tapped: bool, reveal: bool, what: String },
    OptionalPayment { cost: ManaCost, prompt: String, then: Vec<Op> },
    Scry { n: i32 },
    Surveil,
    /// Look at the top n, may take a matching card (Ancient Stirrings).
    /// rest_graveyard: all n are revealed and the rest go to the graveyard (Malevolent Rumble).
    LookTop { filter: SearchFilter, n: i32, what: String, rest_graveyard: bool },
    Cascade,
    Station,
    ReturnRandomFromGraveyard { types: u16 },
    /// types: (type bit, type name) in spec order; any: from either graveyard.
    ReturnCardsFromGraveyards { types: Vec<(u16, &'static str)>, n: i32, each_type: bool, any: bool },
    ExploreTarget,
    ShuffleIntoLibrary,
    /// n +1/+1 counters and keyword counters on the target.
    CountersOnTarget { n: i32, keywords: u32 },
    /// The target becomes a creature with this base P/T and gains keywords.
    AnimateTarget { power: i32, toughness: i32, keywords: u32 },
    TapOrUntapTarget,
    /// Up to n cards of a type from the controller's graveyard to hand; type_name for the prompt.
    ReturnFromGraveyard { types: u16, type_name: String, n: i32 },
    OpponentSacrifices { greatest_power_if_evidence: bool },
    /// Damage equal to the power chosen as the additional cost, to target 0 (Monstrous Emergence).
    DamageChosenPower,
    UntapTarget,
    /// +n/+n until end of turn, or +X/+X with X = the Elves on the battlefield.
    PumpTarget { n: i32, count_elves: bool },
    CountersTarget { n: i32 },
    ShuffleTargetIntoLibrary,
    /// Top n cards: one card type (`take`, or chosen from `choose`) to hand, the rest to the graveyard or the bottom.
    Dig { n: i32, take: Option<String>, choose: Vec<String>, rest_graveyard: bool },
    MayExileFromGraveyard { types: u16, type_name: String, then: Vec<Op> },
    TakeInitiative,
    RevealToBattlefield { n: i32, types: u16, type_name: String, counters: i32, hexproof: bool },
    Custom(Custom),
    /// Engine-internal: the initiative's upkeep venture.
    Venture,
    /// Engine-internal: the ward trigger.
    Ward,
    /// Engine-internal: the madness trigger.
    Madness,
}

impl Op {
    /// The op's name in cards.toml; None for the engine-internal ward and
    /// madness triggers (their Python effects have no ops).
    pub fn name(&self) -> Option<&'static str> {
        Some(match self {
            Op::Draw { .. } => "draw",
            Op::Mill { .. } => "mill",
            Op::CounterTarget { .. } => "counter_target",
            Op::CounterTargetUnlessPaid { .. } => "counter_target_unless_paid",
            Op::DestroyTarget { .. } => "destroy_target",
            Op::BounceTarget => "bounce_target",
            Op::TapTarget { .. } => "tap_target",
            Op::GrantTarget { .. } => "grant_target",
            Op::CreateToken { .. } => "create_token",
            Op::AttachSourceToTarget => "attach_source_to_target",
            Op::GainLife { .. } => "gain_life",
            Op::LoseLife { .. } => "lose_life",
            Op::CounterOnSource => "counter_on_source",
            Op::DamageTarget { .. } => "damage_target",
            Op::DamageEachOpponent { .. } => "damage_each_opponent",
            Op::DamageEachCreature { .. } => "damage_each_creature",
            Op::Discard { .. } => "discard",
            Op::ReturnToBattlefield { .. } => "return_to_battlefield",
            Op::ExileGraveyard => "exile_graveyard",
            Op::ExileAllGraveyards => "exile_all_graveyards",
            Op::SearchLibrary { .. } => "search_library",
            Op::OptionalPayment { .. } => "optional_payment",
            Op::Scry { .. } => "scry",
            Op::ExploreTarget => "explore_target",
            Op::ShuffleIntoLibrary => "shuffle_into_library",
            Op::DamageTargetController { .. } => "damage_target_controller",
            Op::ExileTarget => "exile_target",
            Op::ExileFromGraveyards { .. } => "exile_from_graveyards",
            Op::CountersOnTarget { .. } => "counters_on_target",
            Op::AnimateTarget { .. } => "animate_target",
            Op::TapOrUntapTarget => "tap_or_untap_target",
            Op::ReturnFromGraveyard { .. } => "return_from_graveyard",
            Op::OpponentSacrifices { .. } => "opponent_sacrifices",
            Op::DamageChosenPower => "damage_chosen_power",
            Op::UntapTarget => "untap_target",
            Op::PumpTarget { .. } => "pump_target",
            Op::CountersTarget { .. } => "counters_target",
            Op::ShuffleTargetIntoLibrary => "shuffle_target_into_library",
            Op::Dig { .. } => "dig",
            Op::MayExileFromGraveyard { .. } => "may_exile_from_graveyard",
            Op::TakeInitiative => "take_initiative",
            Op::RevealToBattlefield { .. } => "reveal_to_battlefield",
            Op::DamageTargetFrom { .. } => "damage_target_from",
            Op::Surveil => "surveil",
            Op::LookTop { .. } => "look_top",
            Op::Cascade => "cascade",
            Op::Station => "station",
            Op::ReturnRandomFromGraveyard { .. } => "return_random_from_graveyard",
            Op::ReturnCardsFromGraveyards { .. } => "return_cards_from_graveyards",
            Op::Custom(_) => "custom",
            Op::Ward | Op::Madness | Op::Venture => return None,
        })
    }
}

#[derive(Clone, Debug)]
pub struct AbilityDef {
    pub name: String,
    pub effect: Option<Vec<Op>>,
    pub cost: ManaCost,
    pub tap: bool,
    pub sac_self: bool,
    pub sac_other: Option<SacFilter>,
    pub discard_self: bool,
    pub discard_other: bool,
    pub exile_self: bool,
    /// The cost has this many {X}, X = the target's mana value.
    pub x_target_mv: i32,
    /// X = number of red cards revealed from hand.
    pub x_reveal: bool,
    pub zone_hand: bool,
    pub zone_graveyard: bool,
    pub sorcery_speed: bool,
    pub mana: Option<Vec<u8>>,
    /// One unit per Elf on the battlefield (Priest of Titania).
    pub mana_elves: bool,
    /// Cost: return a Forest you control to its owner's hand (Quirion Ranger).
    pub return_forest: bool,
    pub targets: Vec<TK>,
    /// (n, subtypes): n units while its controller controls each subtype (Urza's Tower).
    pub mana_amount: Option<(i32, Vec<String>)>,
    pub once_per_turn: bool,
    /// Tap another untapped creature you control as a cost (station).
    pub tap_other: bool,
}

impl AbilityDef {
    /// A mana ability with a mana cost ({1}: add one mana of any color).
    pub fn is_filter(&self) -> bool {
        self.mana.is_some() && !self.cost.is_zero()
    }
    pub fn on_battlefield(&self) -> bool {
        !self.zone_hand && !self.zone_graveyard
    }
}

#[derive(Clone, Debug)]
pub struct TriggerDef {
    pub name: String,
    pub event: Event,
    pub effect: Vec<Op>,
    pub sacrificed_subtype: Option<String>,
    pub cast_filter: Option<CastFilter>,
    /// etb condition `{ bargained = true }`.
    pub bargained: bool,
    /// you_cast condition `{ spell = ..., equipped = true }`: only while this
    /// Equipment is attached (a trigger it grants the equipped creature).
    pub equipped: bool,
    /// etb condition `{ entered_untapped = true }`.
    pub entered_untapped: bool,
    /// etb condition `{ cast_mode = ... }`: the cast method's name.
    pub cast_mode: Option<String>,
    /// Chosen as the trigger is put on the stack.
    pub targets: Vec<TK>,
    /// "Up to one target": the targets may be left empty.
    pub up_to: bool,
}

impl TriggerDef {
    pub fn internal(name: &str, event: Event, op: Op) -> TriggerDef {
        TriggerDef { name: name.into(), event, effect: vec![op], sacrificed_subtype: None, cast_filter: None, bargained: false, equipped: false, entered_untapped: false, cast_mode: None, targets: vec![], up_to: false }
    }
}

#[derive(Clone, Debug)]
pub struct SpellMode {
    pub name: String,
    pub targets: Vec<TK>,
    pub effect: Vec<Op>,
}

#[derive(Clone, Debug)]
pub struct CardDef {
    pub id: DefId,
    pub name: String,
    pub cost: ManaCost,
    pub types: u16,
    pub subtypes: Vec<String>,
    pub supertypes: Vec<String>,
    pub colors: u8,
    pub power: Option<i32>,
    pub toughness: Option<i32>,
    pub keywords: u32,
    pub ward: i32,
    pub targets: Vec<TK>,
    pub effect: Option<Vec<Op>>,
    pub additional_sac: Option<SacFilter>,
    pub cost_reduction: Option<CostRed>,
    pub flashback: Option<ManaCost>,
    pub flashback_life: i32,
    pub escape: Option<ManaCost>,
    pub escape_exile: i32,
    pub bestow: Option<ManaCost>,
    pub madness: Option<ManaCost>,
    pub plot: Option<ManaCost>,
    pub overload: Option<ManaCost>,
    pub overload_effect: Option<Vec<Op>>,
    pub additional_discard: bool,
    /// Lands sacrificed instead of mana: (filter, count).
    pub alternative_sac: Option<(SacFilter, i32)>,
    /// Cast mode "alternative": reveal your hand instead of paying the mana
    /// cost, only with no land cards in it (Land Grant).
    pub alternative_reveal: bool,
    pub flashback_sac: Option<(SacFilter, i32)>,
    /// Cast mode "phyrexian": this cost (without the phyrexian symbols) and life.
    pub phyrexian_cost: Option<ManaCost>,
    pub phyrexian_life: i32,
    pub bargain: bool,
    /// Cast mode "evidence": exile cards with this total mana value from the graveyard.
    pub collect_evidence: i32,
    /// Equipment: what the equipped creature gets (Black Mage's Rod: +1/+0).
    pub equipped_power: i32,
    pub equipped_toughness: i32,
    pub abilities: Vec<AbilityDef>,
    pub triggers: Vec<TriggerDef>,
    pub enters_tapped: bool,
    pub etb_x_counters: bool,
    pub back: Option<DefId>,
    pub modes: Vec<SpellMode>,
    /// cards.py `card_shape`: what the card does, as entity tokens read from
    /// its spec (feature set 5). Computed once at load.
    pub shape: Vec<String>,
    pub omen: bool,
    pub enters_tapped_unless_forests: i32,
    pub additional_power: bool,
    pub prototype: Option<ManaCost>,
    pub prototype_face: Option<DefId>,
    /// Spacecraft: a creature with this many charge counters, with these keywords.
    pub station: i32,
    pub station_keywords: u32,
    pub additional_choose_creature: bool,
    /// Equipment: the keywords the equipped creature gets (Whispersilk Cloak).
    pub equipped_keywords: u32,
}

impl CardDef {
    pub fn is_type(&self, t: u16) -> bool {
        self.types & t != 0
    }
    pub fn is_permanent_card(&self) -> bool {
        self.types & (T_ARTIFACT | T_CREATURE | T_ENCHANTMENT | T_LAND | T_PLANESWALKER | T_BATTLE) != 0
    }
    pub fn mana_value(&self) -> i32 {
        self.cost.mana_value()
    }
    pub fn has_subtype(&self, s: &str) -> bool {
        self.subtypes.iter().any(|x| x == s)
    }
    pub fn has_supertype(&self, s: &str) -> bool {
        self.supertypes.iter().any(|x| x == s)
    }
    /// Subtype check that honours changeling (cards.py `has_creature_type`).
    pub fn has_creature_type(&self, s: &str) -> bool {
        self.has_subtype(s) || self.keywords & db().kw("changeling") != 0
    }
}

pub struct CardDb {
    pub defs: Vec<CardDef>,
    pub cards: HashMap<String, DefId>,
    pub tokens: HashMap<String, DefId>,
    pub keyword_names: Vec<String>,
    pub ward: TriggerDef,
    pub madness: TriggerDef,
    pub venture: TriggerDef,
    pub take_initiative: TriggerDef,
    /// The Undercity (its rooms are its triggers) and, per room, the indices of the rooms below it.
    pub undercity: Option<DefId>,
    pub room_next: Vec<Vec<usize>>,
    /// `objects.FREE`: the cost of plot casts and land-sacrifice alternatives.
    pub free: ManaCost,
    pub spec_text: String,
}

/// Keywords the engine itself gives meaning to (or grants).
const ENGINE_KEYWORDS: [&str; 10] = ["changeling", "deathtouch", "flash", "flying", "hexproof", "indestructible", "lifelink", "menace", "reach", "trample"];

impl CardDb {
    pub fn def(&self, id: DefId) -> &CardDef {
        &self.defs[id as usize]
    }

    pub fn kw(&self, name: &str) -> u32 {
        match self.keyword_names.iter().position(|k| k == name) {
            Some(i) => 1 << i,
            None => panic!("unknown keyword {name:?}"),
        }
    }

    pub fn keyword_list(&self, mask: u32) -> Vec<&str> {
        (0..self.keyword_names.len()).filter(|i| mask & (1 << i) != 0).map(|i| self.keyword_names[i].as_str()).collect()
    }

    pub fn parse(text: &str) -> Result<CardDb, String> {
        let root: Table = text.parse::<Table>().map_err(|e| format!("cards.toml: {e}"))?;
        let section = |name: &str| -> Result<Vec<Table>, String> {
            match root.get(name) {
                None => Ok(vec![]),
                Some(Value::Array(a)) => a.iter().map(|v| v.as_table().cloned().ok_or_else(|| format!("[[{name}]] entries must be tables"))).collect(),
                Some(_) => Err(format!("{name} must be an array of tables")),
            }
        };
        let (faces, cards, tokens, dungeons) = (section("face")?, section("card")?, section("token")?, section("dungeon")?);
        for k in root.keys() {
            if !["face", "card", "token", "dungeon"].contains(&k.as_str()) {
                return Err(format!("cards.toml: unknown top-level key {k:?}"));
            }
        }
        // Keyword vocabulary: everything mentioned anywhere, sorted.
        let mut kws: Vec<String> = ENGINE_KEYWORDS.iter().map(|s| s.to_string()).collect();
        fn collect_kws(v: &Value, out: &mut Vec<String>) {
            match v {
                Value::Table(t) => {
                    for (k, x) in t {
                        if k == "keywords" || k == "equipped_keywords" {
                            if let Value::Array(a) = x {
                                for s in a.iter().filter_map(|s| s.as_str()) {
                                    out.push(s.to_string());
                                }
                            }
                        } else if k == "without" {
                            if let Some(s) = x.as_str() {
                                out.push(s.to_string());
                            }
                        } else {
                            collect_kws(x, out);
                        }
                    }
                }
                Value::Array(a) => a.iter().for_each(|x| collect_kws(x, out)),
                _ => {}
            }
        }
        collect_kws(&Value::Table(root.clone()), &mut kws);
        kws.sort();
        kws.dedup();
        if kws.len() > 32 {
            return Err("more than 32 distinct keywords".into());
        }
        let mut db = CardDb {
            defs: vec![],
            cards: HashMap::new(),
            tokens: HashMap::new(),
            keyword_names: kws,
            ward: TriggerDef::internal("ward", Event::BecomesTarget, Op::Ward),
            madness: TriggerDef::internal("madness", Event::Discarded, Op::Madness),
            venture: TriggerDef::internal("venture into Undercity", Event::Initiative, Op::Venture),
            take_initiative: TriggerDef::internal("take the initiative", Event::Initiative, Op::TakeInitiative),
            undercity: None,
            room_next: vec![],
            free: ManaCost::default(),
            spec_text: text.to_string(),
        };
        // Names first so effects can reference tokens and cards can reference faces.
        let mut face_ids = HashMap::new();
        let mut all: Vec<(&str, &Table)> = vec![];
        for (sec, list) in [("face", &faces), ("card", &cards), ("token", &tokens)] {
            for t in list.iter() {
                let name = t.get("name").and_then(|v| v.as_str()).ok_or_else(|| format!("[[{sec}]] entry without a name"))?;
                let id = all.len() as DefId;
                let reg = match sec {
                    "face" => &mut face_ids,
                    "card" => &mut db.cards,
                    _ => &mut db.tokens,
                };
                if reg.insert(name.to_string(), id).is_some() {
                    return Err(format!("duplicate {sec} {name:?}"));
                }
                all.push((sec, t));
            }
        }
        let tokens_by_name = db.tokens.clone();
        for (i, (_, t)) in all.iter().enumerate() {
            let d = parse_card(i as DefId, t, &db, &face_ids, &tokens_by_name).map_err(|e| format!("{}: {e}", t.get("name").and_then(|v| v.as_str()).unwrap_or("?")))?;
            db.defs.push(d);
        }
        for t in &dungeons {
            let (d, next) = parse_dungeon(db.defs.len() as DefId, t, &db, &tokens_by_name).map_err(|e| format!("dungeon: {e}"))?;
            if d.name != "Undercity" {
                return Err(format!("unknown dungeon {:?}", d.name));
            }
            db.undercity = Some(d.id);
            db.room_next = next;
            db.defs.push(d);
        }
        Ok(db)
    }
}

static DB: OnceLock<CardDb> = OnceLock::new();

/// Install the card pool from spec text. Idempotent for identical text.
pub fn load(text: &str) -> Result<(), String> {
    if let Some(db) = DB.get() {
        return if db.spec_text == text { Ok(()) } else { Err("a different card spec is already loaded in this process".into()) };
    }
    let db = CardDb::parse(text)?;
    let _ = DB.set(db);
    Ok(())
}

pub fn db() -> &'static CardDb {
    DB.get_or_init(|| CardDb::parse(BUILTIN_SPEC).expect("built-in cards.toml"))
}

// ---------------------------------------------------------------------------
// Spec parsing helpers
// ---------------------------------------------------------------------------

fn get_str<'a>(t: &'a Table, k: &str) -> Result<Option<&'a str>, String> {
    match t.get(k) {
        None => Ok(None),
        Some(Value::String(s)) => Ok(Some(s.as_str())),
        Some(_) => Err(format!("{k} must be a string")),
    }
}

fn req_str<'a>(t: &'a Table, k: &str) -> Result<&'a str, String> {
    get_str(t, k)?.ok_or_else(|| format!("missing {k}"))
}

fn get_bool(t: &Table, k: &str) -> Result<bool, String> {
    match t.get(k) {
        None => Ok(false),
        Some(Value::Boolean(b)) => Ok(*b),
        Some(_) => Err(format!("{k} must be a boolean")),
    }
}

fn get_int(t: &Table, k: &str) -> Result<Option<i32>, String> {
    match t.get(k) {
        None => Ok(None),
        Some(Value::Integer(i)) => Ok(Some(*i as i32)),
        Some(_) => Err(format!("{k} must be an integer")),
    }
}

fn get_str_list(t: &Table, k: &str) -> Result<Vec<String>, String> {
    match t.get(k) {
        None => Ok(vec![]),
        Some(Value::Array(a)) => a.iter().map(|v| v.as_str().map(|s| s.to_string()).ok_or_else(|| format!("{k} must be a list of strings"))).collect(),
        Some(_) => Err(format!("{k} must be a list")),
    }
}

fn check_keys(t: &Table, allowed: &[&str], what: &str) -> Result<(), String> {
    for k in t.keys() {
        if !allowed.contains(&k.as_str()) {
            return Err(format!("unknown {what} field {k:?}: add it to the allowed keys and the parser here (native/src/cards.rs) and in mtg_ml/engine/cards.py; docs/adding-cards.md"));
        }
    }
    Ok(())
}

/// cards.py `SHAPE_*_FIELDS` / `NON_SHAPE_*_FIELDS`: the spec fields of a
/// card, ability and trigger, split by whether `card_shape` turns them into
/// shape tokens (feature set 5). Both engines list the same fields
/// (`test_spec_field_sets_identical`).
pub const SHAPE_CARD_FIELDS: &[&str] = &[
    "cost", "colors", "devoid", "cost_reduction", "additional_sac", "additional_discard", "flashback", "escape", "madness", "bestow", "plot",
    "overload", "flashback_life", "flashback_cost", "alternative_cost", "ward", "enters_tapped", "etb_x_counters", "back", "targets", "effect", "modes",
    "overload_effect", "abilities", "triggers", "bargain", "collect_evidence", "equipped_power", "equipped_toughness", "equipped_keywords",
    "omen", "enters_tapped_unless_forests", "additional_power", "prototype", "station", "additional_choose_creature",
];
pub const NON_SHAPE_CARD_FIELDS: &[&str] = &["name", "types", "subtypes", "supertypes", "text", "power", "toughness", "keywords", "escape_exile", "prototype_face"];
pub const SHAPE_ABILITY_FIELDS: &[&str] = &[
    "effect", "cost", "tap", "sac_self", "sac_other", "discard_self", "discard_other", "exile_self", "x_target_mv", "x_reveal", "zone", "sorcery_speed", "mana", "targets",
    "mana_amount", "return_land", "once_per_turn", "tap_other",
];
pub const NON_SHAPE_ABILITY_FIELDS: &[&str] = &["name"];
pub const SHAPE_TRIGGER_FIELDS: &[&str] = &["event", "effect", "condition", "targets", "up_to"];
pub const NON_SHAPE_TRIGGER_FIELDS: &[&str] = &["name"];

/// cards.py `_check_fields`: an unclassified field names the lists to extend.
fn check_fields(t: &Table, shape: &[&str], non_shape: &[&str], what: &str) -> Result<(), String> {
    for k in t.keys() {
        if !shape.contains(&k.as_str()) && !non_shape.contains(&k.as_str()) {
            let w = what.to_uppercase();
            return Err(format!(
                "unknown {what} field {k:?}: add it to SHAPE_{w}_FIELDS (if it changes what the card does; then also to card_shape) or NON_SHAPE_{w}_FIELDS, \
                 in both mtg_ml/engine/cards.py and native/src/cards.rs, and parse it in card_def / _ability / _trigger and parse_card; docs/adding-cards.md"
            ));
        }
    }
    Ok(())
}

fn mana(t: &Table, k: &str) -> Result<Option<ManaCost>, String> {
    match get_str(t, k)? {
        None => Ok(None),
        Some(s) => ManaCost::parse(Some(s)).map(Some),
    }
}

fn targets(t: &Table) -> Result<Vec<TK>, String> {
    get_str_list(t, "targets")?.iter().map(|s| TK::parse(s)).collect()
}

fn words(t: &Table, k: &str) -> Result<Vec<String>, String> {
    Ok(get_str(t, k)?.unwrap_or("").split_whitespace().map(|s| s.to_string()).collect())
}

fn parse_ops(v: Option<&Value>, db: &CardDb, tokens: &HashMap<String, DefId>) -> Result<Option<Vec<Op>>, String> {
    let arr = match v {
        None => return Ok(None),
        Some(Value::Array(a)) => a,
        Some(_) => return Err("effect must be a list of ops".into()),
    };
    let mut ops = vec![];
    for v in arr {
        let t = v.as_table().ok_or("ops must be tables")?;
        let op = req_str(t, "op")?;
        let keys: &[&str] = match op {
            "draw" => &["op", "n", "n_cast_from_graveyard", "each_controlling", "who"],
            "mill" => &["op", "who", "n"],
            "counter_target_unless_paid" => &["op", "cost"],
            "tap_target" => &["op", "skip_untap"],
            "grant_target" => &["op", "keywords"],
            "create_token" => &["op", "token", "n", "attach_source"],
            "gain_life" => &["op", "n", "per_storm", "sacrificed_mv", "n_from"],
            "counters_on_target" => &["op", "n", "keywords"],
            "animate_target" => &["op", "power", "toughness", "keywords"],
            "return_from_graveyard" => &["op", "type", "n"],
            "opponent_sacrifices" => &["op", "greatest_power_if_evidence"],
            "scry" | "discard" | "exile_from_graveyards" | "damage_target_controller" | "surveil" => &["op", "n"],
            "damage_target" => &["op", "n", "index", "n_landfall", "n_metalcraft"],
            "damage_target_from" => &["op", "from", "index"],
            "look_top" => &["op", "n", "colorless", "type", "permanent", "rest", "what"],
            "return_random_from_graveyard" => &["op", "type"],
            "return_cards_from_graveyards" => &["op", "types", "n", "each_type", "whose"],
            "damage_each_opponent" => &["op", "n", "if_discarded_nonland"],
            "counter_target" => &["op", "if_color"],
            "destroy_target" => &["op", "if_color", "mv_is_x"],
            "return_to_battlefield" => &["op", "tapped"],
            "lose_life" => &["op", "who", "n"],
            "damage_each_creature" => &["op", "n", "without", "x", "whose", "except_subtype"],
            "search_library" => &["op", "supertype", "type", "subtypes_any", "dest", "tapped", "reveal", "what"],
            "optional_payment" => &["op", "cost", "prompt", "then"],
            "custom" => &["op", "fn"],
            "pump_target" => &["op", "n", "count"],
            "counters_target" => &["op", "n"],
            "dig" => &["op", "n", "take", "choose_type", "rest"],
            "may_exile_from_graveyard" => &["op", "type", "then"],
            "reveal_to_battlefield" => &["op", "n", "type", "counters", "hexproof"],
            _ => &["op"],
        };
        check_keys(t, keys, &format!("op {op:?}"))?;
        let n = || get_int(t, "n")?.ok_or_else(|| format!("op {op:?} needs n"));
        ops.push(match op {
            "draw" => {
                if t.contains_key("who") && t.contains_key("each_controlling") {
                    return Err("draw: who and each_controlling are mutually exclusive".into());
                }
                let target_player = match get_str(t, "who")? {
                    None | Some("you") => false,
                    Some("target_player") => true,
                    Some(w) => return Err(format!("draw: unknown who {w:?}")),
                };
                Op::Draw { n: n()?, n_cast_from_graveyard: get_int(t, "n_cast_from_graveyard")?, each_controlling: get_str(t, "each_controlling")?.map(|s| s.to_string()), target_player }
            },
            "mill" => Op::Mill {
                target_player: match req_str(t, "who")? {
                    "you" => false,
                    "target_player" => true,
                    w => return Err(format!("mill: unknown who {w:?}")),
                },
                n: n()?,
            },
            "counter_target" => Op::CounterTarget { if_color: if_color(t)? },
            "counter_target_unless_paid" => Op::CounterTargetUnlessPaid { cost: ManaCost::parse(Some(req_str(t, "cost")?))? },
            "destroy_target" => Op::DestroyTarget { if_color: if_color(t)?, mv_is_x: get_bool(t, "mv_is_x")? },
            "bounce_target" => Op::BounceTarget,
            "tap_target" => Op::TapTarget { skip_untap: get_int(t, "skip_untap")?.unwrap_or(0) },
            "grant_target" => Op::GrantTarget { keywords: get_str_list(t, "keywords")?.iter().fold(0, |m, k| m | db.kw(k)) },
            "create_token" => {
                let name = req_str(t, "token")?;
                Op::CreateToken {
                    token: *tokens.get(name).ok_or_else(|| format!("unknown token {name:?}"))?,
                    n: get_int(t, "n")?.unwrap_or(1),
                    attach_source: get_bool(t, "attach_source")?,
                }
            }
            "attach_source_to_target" => Op::AttachSourceToTarget,
            "gain_life" => {
                let sacrificed_mv = get_bool(t, "sacrificed_mv")?;
                let source_power = match get_str(t, "n_from")? {
                    None => false,
                    Some("source_power") => true,
                    Some(f) => return Err(format!("gain_life: unknown n_from {f:?}")),
                };
                let n = if sacrificed_mv || source_power { get_int(t, "n")?.unwrap_or(0) } else { n()? };
                Op::GainLife { n, per_storm: get_bool(t, "per_storm")?, sacrificed_mv, source_power }
            }
            "lose_life" => Op::LoseLife {
                who: match req_str(t, "who")? {
                    "you" => Who::You,
                    "opponent" => Who::Opponent,
                    "target_player" => Who::TargetPlayer,
                    "target_controller" => Who::TargetController,
                    w => return Err(format!("lose_life: unknown who {w:?}")),
                },
                n: n()?,
            },
            "counter_on_source" => Op::CounterOnSource,
            "damage_target" => Op::DamageTarget {
                n: n()?,
                index: get_int(t, "index")?.unwrap_or(0).max(0) as usize,
                n_landfall: get_int(t, "n_landfall")?,
                n_metalcraft: get_int(t, "n_metalcraft")?,
            },
            "damage_target_controller" => Op::DamageTargetController { n: n()? },
            "damage_target_from" => Op::DamageTargetFrom {
                index: get_int(t, "index")?.unwrap_or(0).max(0) as usize,
                chosen_power: match req_str(t, "from")? {
                    "x" => false,
                    "chosen_power" => true,
                    f => return Err(format!("damage_target_from: unknown from {f:?}")),
                },
            },
            "damage_each_opponent" => Op::DamageEachOpponent { n: n()?, if_discarded_nonland: get_bool(t, "if_discarded_nonland")? },
            "damage_each_creature" => {
                let x = get_bool(t, "x")?;
                Op::DamageEachCreature {
                    n: if x { get_int(t, "n")?.unwrap_or(0) } else { n()? },
                    x,
                    without: get_str(t, "without")?.map(|k| db.kw(k)).unwrap_or(0),
                    opponent_only: match get_str(t, "whose")? {
                        None => false,
                        Some("opponent") => true,
                        Some(w) => return Err(format!("damage_each_creature: unknown whose {w:?}")),
                    },
                    except_subtype: get_str(t, "except_subtype")?.map(|s| s.to_string()),
                }
            }
            "discard" => Op::Discard { n: n()? },
            "return_to_battlefield" => Op::ReturnToBattlefield { tapped: get_bool(t, "tapped")? },
            "exile_graveyard" => Op::ExileGraveyard,
            "exile_all_graveyards" => Op::ExileAllGraveyards,
            "exile_target" => Op::ExileTarget,
            "exile_from_graveyards" => Op::ExileFromGraveyards { n: n()? },
            "search_library" => Op::SearchLibrary {
                filter: SearchFilter {
                    supertype: get_str(t, "supertype")?.map(|s| s.to_string()),
                    types: match get_str(t, "type")? {
                        Some(s) => type_bit(s)?,
                        None => 0,
                    },
                    subtypes_any: get_str_list(t, "subtypes_any")?,
                    colorless: false,
                },
                to_battlefield: match req_str(t, "dest")? {
                    "battlefield" => true,
                    "hand" => false,
                    d => return Err(format!("search_library: unsupported dest {d:?}")),
                },
                tapped: get_bool(t, "tapped")?,
                reveal: get_bool(t, "reveal")?,
                what: req_str(t, "what")?.to_string(),
            },
            "optional_payment" => Op::OptionalPayment {
                cost: ManaCost::parse(Some(req_str(t, "cost")?))?,
                prompt: req_str(t, "prompt")?.to_string(),
                then: parse_ops(t.get("then"), db, tokens)?.unwrap_or_default(),
            },
            "scry" => Op::Scry { n: n()? },
            "damage_chosen_power" => Op::DamageChosenPower,
            "untap_target" => Op::UntapTarget,
            "pump_target" => match get_str(t, "count")? {
                Some("elves") => Op::PumpTarget { n: 0, count_elves: true },
                Some(c) => return Err(format!("pump_target: unknown count {c:?}")),
                None => Op::PumpTarget { n: n()?, count_elves: false },
            },
            "counters_target" => Op::CountersTarget { n: n()? },
            "shuffle_target_into_library" => Op::ShuffleTargetIntoLibrary,
            "dig" => {
                let take = get_str(t, "take")?.map(|s| s.to_string());
                let choose = get_str_list(t, "choose_type")?;
                for ty in take.iter().chain(choose.iter()) {
                    type_bit(ty)?;
                }
                if take.is_some() == !choose.is_empty() {
                    return Err("dig: exactly one of take and choose_type".into());
                }
                Op::Dig {
                    n: n()?,
                    take,
                    choose,
                    rest_graveyard: match req_str(t, "rest")? {
                        "graveyard" => true,
                        "bottom" => false,
                        r => return Err(format!("dig: unknown rest {r:?}")),
                    },
                }
            }
            "may_exile_from_graveyard" => {
                let ty = req_str(t, "type")?;
                Op::MayExileFromGraveyard { types: type_bit(ty)?, type_name: ty.to_string(), then: parse_ops(t.get("then"), db, tokens)?.unwrap_or_default() }
            }
            "take_initiative" => Op::TakeInitiative,
            "reveal_to_battlefield" => {
                let ty = req_str(t, "type")?;
                Op::RevealToBattlefield { n: n()?, types: type_bit(ty)?, type_name: ty.to_string(), counters: get_int(t, "counters")?.unwrap_or(0), hexproof: get_bool(t, "hexproof")? }
            }
            "surveil" => {
                if n()? != 1 {
                    return Err("surveil: only n = 1 is supported".into());
                }
                Op::Surveil
            }
            "look_top" => Op::LookTop {
                filter: SearchFilter {
                    supertype: None,
                    // `permanent`: any permanent type (the mask matches any of its bits).
                    types: match (get_str(t, "type")?, get_bool(t, "permanent")?) {
                        (Some(_), true) => return Err("look_top: type and permanent are mutually exclusive".into()),
                        (Some(s), false) => type_bit(s)?,
                        (None, true) => T_ARTIFACT | T_BATTLE | T_CREATURE | T_ENCHANTMENT | T_LAND | T_PLANESWALKER,
                        (None, false) => 0,
                    },
                    subtypes_any: vec![],
                    colorless: get_bool(t, "colorless")?,
                },
                n: n()?,
                what: req_str(t, "what")?.to_string(),
                rest_graveyard: match get_str(t, "rest")? {
                    None | Some("bottom") => false,
                    Some("graveyard") => true,
                    Some(r) => return Err(format!("look_top: unknown rest {r:?}")),
                },
            },
            "cascade" => Op::Cascade,
            "station" => Op::Station,
            "return_random_from_graveyard" => Op::ReturnRandomFromGraveyard { types: type_bit(req_str(t, "type")?)? },
            "return_cards_from_graveyards" => Op::ReturnCardsFromGraveyards {
                types: get_str_list(t, "types")?
                    .iter()
                    .map(|s| -> Result<(u16, &'static str), String> {
                        let b = type_bit(s)?;
                        Ok((b, TYPE_NAMES[b.trailing_zeros() as usize]))
                    })
                    .collect::<Result<_, _>>()?,
                n: n()?,
                each_type: get_bool(t, "each_type")?,
                any: match req_str(t, "whose")? {
                    "you" => false,
                    "any" => true,
                    w => return Err(format!("return_cards_from_graveyards: unknown whose {w:?}")),
                },
            },
            "explore_target" => Op::ExploreTarget,
            "shuffle_into_library" => Op::ShuffleIntoLibrary,
            "counters_on_target" => Op::CountersOnTarget { n: get_int(t, "n")?.unwrap_or(0), keywords: get_str_list(t, "keywords")?.iter().fold(0, |m, k| m | db.kw(k)) },
            "animate_target" => Op::AnimateTarget {
                power: get_int(t, "power")?.ok_or("animate_target needs power")?,
                toughness: get_int(t, "toughness")?.ok_or("animate_target needs toughness")?,
                keywords: get_str_list(t, "keywords")?.iter().fold(0, |m, k| m | db.kw(k)),
            },
            "tap_or_untap_target" => Op::TapOrUntapTarget,
            "return_from_graveyard" => {
                let ty = req_str(t, "type")?;
                Op::ReturnFromGraveyard { types: type_bit(ty)?, type_name: ty.to_lowercase(), n: n()? }
            }
            "opponent_sacrifices" => Op::OpponentSacrifices { greatest_power_if_evidence: get_bool(t, "greatest_power_if_evidence")? },
            "custom" => Op::Custom(match req_str(t, "fn")? {
                "delver_reveal" => Custom::DelverReveal,
                "brainstorm" => Custom::Brainstorm,
                "ponder" => Custom::Ponder,
                "deem_inferior" => Custom::DeemInferior,
                "opponent_discards_else_draw" => Custom::OpponentDiscardsElseDraw,
                "wildfire" => Custom::Wildfire,
                "duress" => Custom::Duress,
                "highway_robbery" => Custom::HighwayRobbery,
                "relic_exile_one" => Custom::RelicExileOne,
                f => return Err(format!("unknown custom effect {f:?}: add it to Custom (native/src/cards.rs) and Eng::custom (native/src/engine.rs)")),
            }),
            _ => return Err(format!("unknown op {op:?}: add it to Op (native/src/cards.rs) and Eng::run_op (native/src/engine.rs)")),
        });
    }
    Ok(Some(ops))
}

/// `if_color = "U"`: the colour bit, 0 when absent.
fn if_color(t: &Table) -> Result<u8, String> {
    match get_str(t, "if_color")? {
        None => Ok(0),
        Some(c) if c.len() == 1 && color_bit(c.as_bytes()[0]) != 0 => Ok(color_bit(c.as_bytes()[0])),
        Some(c) => Err(format!("unknown if_color {c:?}")),
    }
}

/// `{ sacrifice = "mountain", n = N }`: lands sacrificed instead of mana.
/// `{ reveal_hand = true }`: reveal your hand instead of paying (Land Grant).
fn reveal_hand(t: &Table, k: &str) -> bool {
    matches!(t.get(k), Some(Value::Table(v)) if v.len() == 1 && v.get("reveal_hand") == Some(&Value::Boolean(true)))
}

fn land_sac(t: &Table, k: &str) -> Result<Option<(SacFilter, i32)>, String> {
    let v = match t.get(k) {
        None => return Ok(None),
        Some(Value::Table(v)) => v,
        Some(_) => return Err(format!("{k} must be a table")),
    };
    if reveal_hand(t, k) {
        return Ok(None);
    }
    if v.len() != 2 || req_str(v, "sacrifice")? != "mountain" {
        return Err(format!("unsupported land sacrifice cost {v}"));
    }
    let n = get_int(v, "n")?.ok_or_else(|| format!("unsupported land sacrifice cost {v}"))?;
    Ok(Some((SacFilter::Mountain, n)))
}

fn parse_card(id: DefId, t: &Table, db: &CardDb, faces: &HashMap<String, DefId>, tokens: &HashMap<String, DefId>) -> Result<CardDef, String> {
    check_fields(t, SHAPE_CARD_FIELDS, NON_SHAPE_CARD_FIELDS, "card")?;
    let subtypes = words(t, "subtypes")?;
    let equipped_keywords = get_str_list(t, "equipped_keywords")?.iter().fold(0, |m, k| m | db.kw(k));
    let cost_text = get_str(t, "cost")?;
    let colors = match get_str(t, "colors")? {
        Some(c) => c.split_whitespace().fold(0, |m, s| m | color_bit(s.as_bytes()[0])),
        None if get_bool(t, "devoid")? || cost_text.map_or(true, |c| c.is_empty()) => 0,
        None => [b'W', b'U', b'B', b'R', b'G']
            .iter()
            .filter(|c| cost_text.unwrap().contains(&format!("{{{}}}", **c as char)) || cost_text.unwrap().contains(&format!("{{{}/P}}", **c as char)))
            .fold(0, |m, c| m | color_bit(*c)),
    };
    let mut types = 0;
    for w in words(t, "types")? {
        types |= type_bit(&w)?;
    }
    let mut abilities = vec![];
    if let Some(v) = t.get("abilities") {
        for a in v.as_array().ok_or("abilities must be a list")? {
            let a = a.as_table().ok_or("abilities must be tables")?;
            check_fields(a, SHAPE_ABILITY_FIELDS, NON_SHAPE_ABILITY_FIELDS, "ability")?;
            if a.contains_key("mana") && a.contains_key("cost") && mana(a, "cost")?.unwrap_or_default() != ManaCost::generic(1) {
                return Err(format!("{}: a mana filter must cost exactly {{1}}", req_str(a, "name")?));
            }
            let mana_amount = match a.get("mana_amount") {
                None => None,
                Some(Value::Table(m)) if m.len() == 2 => Some((get_int(m, "n")?.ok_or("mana_amount needs n")?, get_str_list(m, "if_control")?)),
                Some(Value::String(e)) if e == "elves" => None, // mana_elves
                Some(m) => return Err(format!("unsupported mana_amount {m}")),
            };
            let x_target_mv = get_int(a, "x_target_mv")?.unwrap_or(0);
            if x_target_mv != 0 && targets(a)?.len() != 1 {
                return Err("x_target_mv needs exactly one target".into());
            }
            abilities.push(AbilityDef {
                name: req_str(a, "name")?.to_string(),
                effect: parse_ops(a.get("effect"), db, tokens)?,
                cost: mana(a, "cost")?.unwrap_or_default(),
                tap: get_bool(a, "tap")?,
                sac_self: get_bool(a, "sac_self")?,
                sac_other: get_str(a, "sac_other")?.map(SacFilter::parse).transpose()?,
                discard_self: get_bool(a, "discard_self")?,
                discard_other: get_bool(a, "discard_other")?,
                exile_self: get_bool(a, "exile_self")?,
                x_target_mv,
                x_reveal: match get_str(a, "x_reveal")? {
                    None => false,
                    Some("red") => true,
                    Some(r) => return Err(format!("unknown x_reveal {r:?}")),
                },
                zone_hand: match get_str(a, "zone")?.unwrap_or("battlefield") {
                    "battlefield" | "graveyard" => false,
                    "hand" => true,
                    z => return Err(format!("unknown ability zone {z:?}")),
                },
                zone_graveyard: get_str(a, "zone")? == Some("graveyard"),
                sorcery_speed: get_bool(a, "sorcery_speed")?,
                mana: if a.contains_key("mana") { Some(get_str_list(a, "mana")?.iter().map(|s| s.as_bytes()[0]).collect()) } else { None },
                mana_elves: matches!(a.get("mana_amount"), Some(Value::String(e)) if e == "elves"),
                return_forest: match get_str(a, "return_land")? {
                    None => false,
                    Some("forest") => true,
                    Some(r) => return Err(format!("unknown return_land {r:?}")),
                },
                targets: targets(a)?,
                mana_amount,
                once_per_turn: get_bool(a, "once_per_turn")?,
                tap_other: match get_str(a, "tap_other")? {
                    None => false,
                    Some("creature") => true,
                    Some(o) => return Err(format!("unknown tap_other {o:?}")),
                },
            });
        }
    }
    let mut triggers = vec![];
    if let Some(v) = t.get("triggers") {
        for tr in v.as_array().ok_or("triggers must be a list")? {
            let tr = tr.as_table().ok_or("triggers must be tables")?;
            check_fields(tr, SHAPE_TRIGGER_FIELDS, NON_SHAPE_TRIGGER_FIELDS, "trigger")?;
            let (mut sacrificed_subtype, mut cast_filter, mut bargained, mut equipped, mut entered_untapped, mut cast_mode) = (None, None, false, false, false, None);
            let spell_filter = |c: &Table| -> Result<CastFilter, String> {
                Ok(match req_str(c, "spell")? {
                    "noncreature" => CastFilter::Noncreature,
                    "instant_or_sorcery" => CastFilter::InstantOrSorcery,
                    f => return Err(format!("unknown spell condition {f:?}")),
                })
            };
            match tr.get("condition") {
                None => {}
                Some(Value::Table(c)) if c.len() == 1 && c.get("bargained") == Some(&Value::Boolean(true)) => bargained = true,
                Some(Value::Table(c)) if c.len() == 1 && c.get("entered_untapped") == Some(&Value::Boolean(true)) => entered_untapped = true,
                Some(Value::Table(c)) if c.len() == 1 && c.contains_key("cast_mode") => cast_mode = Some(req_str(c, "cast_mode")?.to_string()),
                Some(Value::Table(c)) if c.len() == 1 && c.contains_key("sacrificed_subtype") => sacrificed_subtype = Some(req_str(c, "sacrificed_subtype")?.to_string()),
                Some(Value::Table(c)) if c.len() == 1 && c.contains_key("spell") => cast_filter = Some(spell_filter(c)?),
                Some(Value::Table(c)) if c.len() == 2 && c.contains_key("spell") && c.get("equipped") == Some(&Value::Boolean(true)) => {
                    cast_filter = Some(spell_filter(c)?);
                    equipped = true;
                }
                Some(c) => return Err(format!("unsupported trigger condition {c}")),
            }
            triggers.push(TriggerDef {
                name: req_str(tr, "name")?.to_string(),
                event: match req_str(tr, "event")? {
                    "etb" => Event::Etb,
                    "to_graveyard_from_battlefield" => Event::ToGraveyardFromBattlefield,
                    "cast" => Event::Cast,
                    "you_sacrifice_another" => Event::YouSacrificeAnother,
                    "your_upkeep" => Event::YourUpkeep,
                    "you_cast" => Event::YouCast,
                    "third_draw" => Event::ThirdDraw,
                    "leaves_battlefield" => Event::LeavesBattlefield,
                    e => return Err(format!("unknown trigger event {e:?}")),
                },
                effect: parse_ops(tr.get("effect"), db, tokens)?.ok_or("trigger without effect")?,
                sacrificed_subtype,
                cast_filter,
                bargained,
                equipped,
                entered_untapped,
                cast_mode,
                targets: targets(tr)?,
                up_to: get_bool(tr, "up_to")?,
            });
        }
    }
    let mut modes = vec![];
    if let Some(v) = t.get("modes") {
        for m in v.as_array().ok_or("modes must be a list")? {
            let m = m.as_table().ok_or("modes must be tables")?;
            check_keys(m, &["name", "targets", "effect"], "mode")?;
            modes.push(SpellMode {
                name: req_str(m, "name")?.to_string(),
                targets: targets(m)?,
                effect: parse_ops(m.get("effect"), db, tokens)?.ok_or("mode without effect")?,
            });
        }
    }
    let phy = ManaCost::phyrexian(cost_text);
    let cost = ManaCost::parse(cost_text)?;
    let shape = card_shape(t, &cost, colors)?;
    Ok(CardDef {
        id,
        name: req_str(t, "name")?.to_string(),
        phyrexian_cost: if phy.colored.is_empty() { None } else { Some(cost.minus_colored(&phy)) },
        phyrexian_life: 2 * phy.mana_value(),
        flashback_life: get_int(t, "flashback_life")?.unwrap_or(0),
        bargain: get_bool(t, "bargain")?,
        cost,
        types,
        subtypes,
        supertypes: words(t, "supertypes")?,
        colors,
        power: get_int(t, "power")?,
        toughness: get_int(t, "toughness")?,
        keywords: get_str_list(t, "keywords")?.iter().fold(0, |m, k| m | db.kw(k)),
        ward: get_int(t, "ward")?.unwrap_or(0),
        targets: targets(t)?,
        effect: parse_ops(t.get("effect"), db, tokens)?,
        additional_sac: get_str(t, "additional_sac")?.map(SacFilter::parse).transpose()?,
        cost_reduction: match get_str(t, "cost_reduction")? {
            None => None,
            Some("instants_and_sorceries_in_graveyard") => Some(CostRed::InstantsAndSorceriesInGraveyard),
            Some("artifacts_you_control") => Some(CostRed::ArtifactsYouControl),
            Some("cards_drawn_this_turn") => Some(CostRed::CardsDrawnThisTurn),
            Some(c) => return Err(format!("unknown cost_reduction {c:?}")),
        },
        flashback: match mana(t, "flashback")? {
            Some(m) => Some(m),
            None if t.contains_key("flashback_cost") => Some(ManaCost::default()),
            None => None,
        },
        escape: mana(t, "escape")?,
        escape_exile: get_int(t, "escape_exile")?.unwrap_or(0),
        bestow: mana(t, "bestow")?,
        madness: mana(t, "madness")?,
        plot: mana(t, "plot")?,
        overload: mana(t, "overload")?,
        overload_effect: parse_ops(t.get("overload_effect"), db, tokens)?,
        additional_discard: get_bool(t, "additional_discard")?,
        alternative_sac: land_sac(t, "alternative_cost")?,
        alternative_reveal: reveal_hand(t, "alternative_cost"),
        flashback_sac: land_sac(t, "flashback_cost")?,
        collect_evidence: get_int(t, "collect_evidence")?.unwrap_or(0),
        equipped_power: get_int(t, "equipped_power")?.unwrap_or(0),
        equipped_toughness: get_int(t, "equipped_toughness")?.unwrap_or(0),
        abilities,
        triggers,
        enters_tapped: get_bool(t, "enters_tapped")?,
        etb_x_counters: get_bool(t, "etb_x_counters")?,
        back: match get_str(t, "back")? {
            None => None,
            Some(b) => Some(*faces.get(b).ok_or_else(|| format!("unknown back face {b:?}"))?),
        },
        modes,
        shape,
        omen: get_bool(t, "omen")?,
        enters_tapped_unless_forests: get_int(t, "enters_tapped_unless_forests")?.unwrap_or(0),
        additional_power: get_bool(t, "additional_power")?,
        prototype: mana(t, "prototype")?,
        prototype_face: match get_str(t, "prototype_face")? {
            None => None,
            Some(b) => Some(*faces.get(b).ok_or_else(|| format!("unknown prototype face {b:?}"))?),
        },
        station: match t.get("station") {
            None => 0,
            Some(Value::Table(st)) => {
                check_keys(st, &["n", "keywords"], "station")?;
                get_int(st, "n")?.ok_or("station needs n")?
            }
            Some(_) => return Err("station must be a table".into()),
        },
        station_keywords: match t.get("station") {
            Some(Value::Table(st)) => get_str_list(st, "keywords")?.iter().fold(0, |m, k| m | db.kw(k)),
            _ => 0,
        },
        additional_choose_creature: get_bool(t, "additional_choose_creature")?,
        equipped_keywords,
    })
}

/// cards.py `SHAPE_MV_STEPS` / `SHAPE_AB_MV_CAP` / `SHAPE_COST_KEYS`.
const SHAPE_MV_STEPS: [i32; 7] = [1, 2, 3, 4, 5, 6, 7];
const SHAPE_AB_MV_CAP: i32 = 3;
const SHAPE_COST_KEYS: [&str; 6] = ["flashback", "escape", "madness", "bestow", "plot", "overload"];

fn tables(v: Option<&Value>) -> Vec<&Table> {
    match v {
        Some(Value::Array(a)) => a.iter().filter_map(|x| x.as_table()).collect(),
        _ => vec![],
    }
}

/// cards.py `SHAPE_N_STEPS`.
const SHAPE_N_STEPS: [i64; 4] = [1, 2, 3, 4];
/// cards.py `SHAPE_OP_FLAGS`.
const SHAPE_OP_FLAGS: [&str; 6] = ["n_metalcraft", "sacrificed_mv", "greatest_power_if_evidence", "attach_source", "n_from", "each_type"];

/// cards.py `_op_tokens`: `{prefix}{op}` per op in order, its amount `n` as
/// a thermometer, the same under `e:op:`, an `optional_payment`'s ops after it.
fn op_names(v: Option<&Value>, out: &mut Vec<String>, prefix: &str) {
    for op in tables(v) {
        if let Some(name) = op.get("op").and_then(|x| x.as_str()) {
            for p in [prefix, "e:op:"] {
                out.push(format!("{p}{name}"));
                if let Some(Value::Integer(n)) = op.get("n") {
                    out.extend(SHAPE_N_STEPS.iter().filter(|&&k| *n >= k).map(|k| format!("{p}{name}:n>={k}")));
                }
                out.extend(SHAPE_OP_FLAGS.iter().filter(|&&k| truthy(op, k)).map(|k| format!("{p}{name}:{k}")));
            }
            if name == "optional_payment" {
                op_names(op.get("then"), out, prefix);
            }
        }
    }
}

fn truthy(t: &Table, k: &str) -> bool {
    match t.get(k) {
        None | Some(Value::Boolean(false)) => false,
        Some(Value::Integer(0)) => false,
        Some(Value::String(s)) => !s.is_empty(),
        Some(_) => true,
    }
}

/// cards.py `card_shape`: the same token list from the same spec table.
fn card_shape(t: &Table, cost: &ManaCost, colors: u8) -> Result<Vec<String>, String> {
    let mut v: Vec<String> = vec![];
    let mv = cost.mana_value();
    v.extend(SHAPE_MV_STEPS.iter().filter(|&&k| mv >= k).map(|k| format!("e:mv>={k}")));
    if get_str(t, "cost")?.unwrap_or("").contains("{X}") {
        v.push("e:cost:x".into());
    }
    if get_str(t, "cost")?.unwrap_or("").contains("/P}") {
        v.push("e:cost:phyrexian".into());
    }
    for c in [b'W', b'U', b'B', b'R', b'G'] {
        if colors & color_bit(c) != 0 {
            v.push(format!("e:color:{}", c as char));
        }
    }
    if let Some(r) = get_str(t, "cost_reduction")? {
        v.push(format!("e:cost:reduction:{r}"));
    }
    if let Some(s) = get_str(t, "additional_sac")? {
        v.push("e:cost:additional_sac".into());
        v.push(format!("e:cost:additional_sac:{s}"));
    }
    if truthy(t, "additional_discard") {
        v.push("e:cost:additional_discard".into());
    }
    if truthy(t, "bargain") {
        v.push("e:cost:bargain".into());
    }
    if truthy(t, "collect_evidence") {
        v.push("e:cost:collect_evidence".into());
    }
    if truthy(t, "additional_power") {
        v.push("e:cost:additional_power".into());
    }
    if truthy(t, "omen") {
        v.push("e:cost:omen".into());
    }
    if truthy(t, "equipped_power") || truthy(t, "equipped_toughness") || truthy(t, "equipped_keywords") {
        v.push("e:equipment_bonus".into());
    }
    for k in get_str_list(t, "equipped_keywords")? {
        v.push(format!("e:equipped:kw:{k}"));
    }
    if t.contains_key("prototype") {
        v.push("e:cost:prototype".into());
    }
    if truthy(t, "additional_choose_creature") {
        v.push("e:cost:additional_choose_creature".into());
    }
    if let Some(Value::Table(st)) = t.get("station") {
        v.push("e:station".into());
        for k in get_str_list(st, "keywords")? {
            v.push(format!("e:station:kw:{k}"));
        }
    }
    for k in SHAPE_COST_KEYS {
        if t.contains_key(k) {
            v.push(format!("e:cost:{k}"));
        }
    }
    if truthy(t, "flashback_life") {
        v.push("e:cost:flashback_life".into());
    }
    if t.contains_key("flashback_cost") {
        v.push("e:cost:flashback".into());
        v.push("e:cost:sac_lands".into());
    }
    if t.contains_key("alternative_cost") {
        v.push("e:cost:alternative".into());
        v.push(if reveal_hand(t, "alternative_cost") { "e:cost:reveal_hand" } else { "e:cost:sac_lands" }.into());
    }
    if truthy(t, "ward") {
        v.push("e:ward".into());
    }
    for k in ["enters_tapped", "etb_x_counters", "enters_tapped_unless_forests"] {
        if truthy(t, k) {
            v.push(format!("e:{k}"));
        }
    }
    if t.contains_key("back") {
        v.push("e:transforms".into());
    }
    // cards.py `_target_tokens`.
    let targets = |t: &Table, prefix: &str, v: &mut Vec<String>| -> Result<(), String> {
        for k in get_str_list(t, "targets")? {
            v.push(format!("{prefix}{k}"));
            v.push(format!("e:target:{k}"));
        }
        Ok(())
    };
    targets(t, "e:spell:target:", &mut v)?;
    op_names(t.get("effect"), &mut v, "e:spell:op:");
    let modes = tables(t.get("modes"));
    if !modes.is_empty() {
        v.push("e:spell:modal".into());
    }
    for m in modes {
        targets(m, "e:spell:target:", &mut v)?;
        op_names(m.get("effect"), &mut v, "e:spell:op:");
    }
    op_names(t.get("overload_effect"), &mut v, "e:spell:op:");
    for a in tables(t.get("abilities")) {
        v.push(format!("e:ab:zone:{}", get_str(a, "zone")?.unwrap_or("battlefield")));
        if a.contains_key("mana") {
            v.push("e:ab:mana".into());
            for c in get_str_list(a, "mana")? {
                v.push(format!("e:ab:mana:{c}"));
            }
        }
        // "elves" (a count) or Tron's { n, if_control }.
        match a.get("mana_amount") {
            Some(Value::String(m)) => v.push(format!("e:ab:mana_amount:{m}")),
            Some(Value::Table(m)) => v.push(format!("e:ab:mana_amount:{}", get_int(m, "n")?.unwrap_or(0))),
            _ => {}
        }
        let mv = ManaCost::parse(get_str(a, "cost")?)?.mana_value();
        if mv > 0 {
            v.push(format!("e:ab:mv:{}", mv.min(SHAPE_AB_MV_CAP)));
        }
        for k in ["tap", "sac_self", "sac_other", "discard_self", "discard_other", "exile_self"] {
            if truthy(a, k) {
                v.push(format!("e:ab:{k}"));
            }
        }
        if let Some(s) = get_str(a, "sac_other")? {
            v.push(format!("e:ab:sac_other:{s}"));
        }
        if let Some(r) = get_str(a, "return_land")? {
            v.push(format!("e:ab:return_land:{r}"));
        }
        if truthy(a, "once_per_turn") {
            v.push("e:ab:once_per_turn".into());
        }
        if truthy(a, "x_target_mv") || a.contains_key("x_reveal") {
            v.push("e:ab:x".into());
        }
        if truthy(a, "sorcery_speed") {
            v.push("e:ab:sorcery_speed".into());
        }
        if let Some(s) = get_str(a, "tap_other")? {
            v.push(format!("e:ab:tap_other:{s}"));
        }
        targets(a, "e:ab:target:", &mut v)?;
        op_names(a.get("effect"), &mut v, "e:ab:op:");
    }
    for tr in tables(t.get("triggers")) {
        v.push(format!("e:trig:{}", req_str(tr, "event")?));
        if let Some(Value::Table(c)) = tr.get("condition") {
            for (k, x) in c {
                // cards.py: strings as they are, other values lowercased (`true`).
                let x = match x {
                    Value::String(s) => s.clone(),
                    other => other.to_string(),
                };
                v.push(format!("e:trig:cond:{k}:{x}"));
            }
        }
        targets(tr, "e:trig:target:", &mut v)?;
        if truthy(tr, "up_to") {
            v.push("e:trig:up_to".into());
        }
        op_names(tr.get("effect"), &mut v, "e:trig:op:");
    }
    let mut seen = std::collections::HashSet::new();
    v.retain(|x| seen.insert(x.clone()));
    Ok(v)
}

/// `[[dungeon]]`: name and rooms (name, next, targets, effect), the first
/// room on top. The rooms become the dungeon's triggers (event room).
fn parse_dungeon(id: DefId, t: &Table, db: &CardDb, tokens: &HashMap<String, DefId>) -> Result<(CardDef, Vec<Vec<usize>>), String> {
    check_keys(t, &["name", "rooms"], "dungeon")?;
    let rooms = t.get("rooms").and_then(|v| v.as_array()).ok_or("rooms must be a list")?;
    let mut names: Vec<String> = vec![];
    for r in rooms {
        let r = r.as_table().ok_or("rooms must be tables")?;
        names.push(req_str(r, "name")?.to_string());
    }
    let mut triggers = vec![];
    let mut next = vec![];
    for r in rooms {
        let r = r.as_table().unwrap();
        check_keys(r, &["name", "next", "targets", "effect"], "room")?;
        let mut nx = vec![];
        for n in get_str_list(r, "next")? {
            nx.push(names.iter().position(|x| *x == n).ok_or_else(|| format!("unknown next room {n:?}"))?);
        }
        next.push(nx);
        triggers.push(TriggerDef {
            name: req_str(r, "name")?.to_string(),
            event: Event::Room,
            effect: parse_ops(r.get("effect"), db, tokens)?.ok_or("room without effect")?,
            sacrificed_subtype: None,
            cast_filter: None,
            bargained: false,
            equipped: false,
            entered_untapped: false,
            cast_mode: None,
            targets: targets(r)?,
            up_to: false,
        });
    }
    let d = CardDef {
        id,
        name: req_str(t, "name")?.to_string(),
        cost: ManaCost::default(),
        types: 0,
        subtypes: vec![],
        supertypes: vec![],
        colors: 0,
        power: None,
        toughness: None,
        keywords: 0,
        ward: 0,
        targets: vec![],
        effect: None,
        additional_sac: None,
        cost_reduction: None,
        flashback: None,
        flashback_life: 0,
        escape: None,
        escape_exile: 0,
        bestow: None,
        madness: None,
        plot: None,
        overload: None,
        overload_effect: None,
        additional_discard: false,
        alternative_sac: None,
        alternative_reveal: false,
        flashback_sac: None,
        abilities: vec![],
        triggers,
        enters_tapped: false,
        etb_x_counters: false,
        back: None,
        modes: vec![],
        omen: false,
        enters_tapped_unless_forests: 0,
        additional_power: false,
        collect_evidence: 0,
        equipped_power: 0,
        equipped_toughness: 0,
        equipped_keywords: 0,
        prototype: None,
        prototype_face: None,
        station: 0,
        station_keywords: 0,
        additional_choose_creature: false,
        shape: vec![],
        bargain: false,
        phyrexian_cost: None,
        phyrexian_life: 0,
    };
    Ok((d, next))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builtin_spec_parses() {
        let db = db();
        assert!(db.cards.contains_key("Brainstorm"));
        let delver = db.def(db.cards["Delver of Secrets"]);
        let back = db.def(delver.back.unwrap());
        assert_eq!(back.name, "Insectile Aberration");
        assert_eq!(back.colors, color_bit(b'U'));
        assert_eq!(db.def(db.cards["Writhing Chrysalis"]).colors, 0);
        assert_eq!(db.cards.len(), 134);
        assert!(db.cards.contains_key("Murmuring Mystic"));
        assert!(db.tokens.contains_key("Bird Illusion"));
        let analysis = db.def(db.cards["Deep Analysis"]);
        assert_eq!(analysis.flashback_life, 3);
        assert_eq!(analysis.flashback.as_ref().unwrap().mana_value(), 2);
        assert_eq!(analysis.targets, vec![TK::Player]);
        assert!(db.undercity.is_some() && db.room_next[0] == vec![1, 2]);
        let gut = db.def(db.cards["Gut Shot"]);
        assert_eq!((gut.colors, gut.phyrexian_life, gut.cost.mana_value()), (color_bit(b'R'), 2, 1));
        assert!(gut.phyrexian_cost.as_ref().unwrap().is_zero());
        let blaze = db.def(db.cards["Searing Blaze"]);
        assert_eq!(blaze.targets, vec![TK::PlayerWithCreature, TK::CreatureOfTargetPlayer]);
        assert_eq!(db.def(db.cards["Lava Dart"]).flashback_sac, Some((SacFilter::Mountain, 1)));
        assert!(db.def(db.cards["Lava Dart"]).flashback.as_ref().unwrap().is_zero());
    }
}
