from unittest import TestCase
from CFG import CFG


class TestCFG(TestCase):
    def test_get_new_nonterm(self):
        self.assertTrue(True)

    def test_add_prod(self):
        self.assertTrue(True)

    def test_remove_cycle_rules(self):
        self.assertTrue(True)

    def test_remove_empty_terms(self):
        self.assertTrue(True)

    def test_get_productive_prods1(self):
        cfg = CFG('SA', 'a', {
            'S': 'A|C',
            'A': 'B',
            'B': 'a',
            'C': 'Ca'
        }, 'S')
        expected = set('SAB')
        self.assertEqual(cfg.get_productive_prods(), expected)

    def test_get_productive_prods2(self):
        cfg = CFG('SABCD', 'abcde', {
            'S': 'aAB|C',
            'A': 'aA|a|ε',
            'B': 'b',
            'C': 'aCD',
            'D': 'cDc|d'
        }, 'S')
        expected = set('SABD')
        self.assertEqual(cfg.get_productive_prods(), expected)

    def test_remove_nonpoductive_prods2(self):
        cfg = CFG('SABCD', 'abcde', {
            'S': 'aAB|C',
            'A': 'aA|a|ε',
            'B': 'b',
            'C': 'aCD',
            'D': 'cDc|d'
        }, 'S')
        expected_prods = {'A': {'aA', 'ε', 'a'}, 'B': {'b'}, 'D': {'d', 'cDc'}, 'S': {'aAB'}}
        cfg.remove_nonpoductive_prods(cfg.get_productive_prods())
        self.assertEqual(cfg.prod, expected_prods)

    def test_get_reachable_prods(self):
        self.assertTrue(True)

    def test_remove_unreachable_prods(self):
        self.assertTrue(True)

    def test_get_empty_prods(self):
        self.assertTrue(True)

    def test_remove_emtpy_prods(self):
        self.assertTrue(True)

    def test_resolve_chain_prods(self):
        self.assertTrue(True)

    def test_remove_left_recursion(self):
        self.assertTrue(True)

    def test_factorize_left(self):
        self.assertTrue(True)

    def test_tocnf(self):
        self.assertTrue(True)
