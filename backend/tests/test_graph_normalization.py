from tools.entity_resolver import expand_abbreviations, fuzzy_resolve


def test_expand_lowercases_cox_to_cyclooxygenase():
    cypher = "MATCH (p:Protein) WHERE toLower(p.name) CONTAINS 'cox' RETURN p LIMIT 5"
    out, changed = expand_abbreviations(cypher)
    assert changed
    assert "'cyclooxygenase'" in out
    assert "'cox'" not in out


def test_expand_handles_cox_2_variant():
    out, changed = expand_abbreviations("WHERE p.name CONTAINS 'COX-2'")
    assert changed
    assert out == "WHERE p.name CONTAINS 'cyclooxygenase'"


def test_expand_token_inside_multiword_literal():
    out, changed = expand_abbreviations("WHERE p.name CONTAINS 'COX inhibitors'")
    assert changed
    assert out == "WHERE p.name CONTAINS 'cyclooxygenase inhibitors'"


def test_expand_leaves_normal_compound_names_untouched():
    cypher = "MATCH (c:Compound {name: 'Aspirin'}) RETURN c LIMIT 1"
    out, changed = expand_abbreviations(cypher)
    assert not changed
    assert out == cypher


def test_expand_respects_word_boundaries():
    cypher = "WHERE d.name CONTAINS 'coxsackievirus'"
    out, changed = expand_abbreviations(cypher)
    assert not changed
    assert out == cypher


def test_expand_only_touches_literals_not_structure():
    cypher = "MATCH (cox:Protein) RETURN cox.name LIMIT 5"
    out, changed = expand_abbreviations(cypher)
    assert not changed
    assert out == cypher


NAMES = ["Cyclooxygenase", "Prostaglandin G/H synthase 1", "Carbonic anhydrase 2"]


def test_fuzzy_corrects_a_spelling_variant():
    cypher = "WHERE toLower(p.name) CONTAINS 'cyclooxigenase'"
    out, subs = fuzzy_resolve(cypher, names=NAMES)
    assert subs == [("cyclooxigenase", "Cyclooxygenase")]
    assert "'cyclooxygenase'" in out
    assert "'Cyclooxygenase'" not in out


def test_fuzzy_skips_exact_case_insensitive_match():
    cypher = "WHERE toLower(p.name) CONTAINS 'cyclooxygenase'"
    out, subs = fuzzy_resolve(cypher, names=NAMES)
    assert subs == []
    assert out == cypher


def test_fuzzy_leaves_distant_literals_alone():
    cypher = "MATCH (c:Compound {name: 'Aspirin'}) RETURN c LIMIT 1"
    out, subs = fuzzy_resolve(cypher, names=NAMES)
    assert subs == []
    assert out == cypher


def test_fuzzy_noop_when_graph_has_no_entities():
    cypher = "WHERE p.name CONTAINS 'whatever'"
    out, subs = fuzzy_resolve(cypher, names=[])
    assert subs == []
    assert out == cypher
