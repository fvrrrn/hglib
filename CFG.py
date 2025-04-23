import copy
from utils import *


class CFG(object):
    def __init__(self, p, n=None, v=None, s=None):
        if s is None:
            self.S = p[0][0]
        else:
            self.S = s
        if n is None:
            self.nonterm = set()
        else:
            self.nonterm = set(n)
        if v is None:
            self.term = set()
        else:
            self.term = set(v)
        self.term.add('ε')
        self.prod = {}
        for production in p:
            production = production.replace(' ', '')
            tmp = production.split('→')
            lhs = tmp[0]
            rhs = tmp[1]
            # for lhs, rhs in p.items():
            self.prod[lhs] = set()
            self.nonterm.add(lhs)
            for part in rhs.split('|'):
                if len(part) > 0:
                    self.prod[lhs].add(part)
                    for p in part:
                        if p == p.lower():
                            self.term.add(p)
                            continue
                        self.nonterm.add(p)

    def get_new_nonterm(self, nonterm=None):
        if nonterm is None:
            nonterm = self.nonterm
        for c in sorted(set('ABCDEFGHIJKLMNOPQRSTUVWXYZ').difference(nonterm)):
            return c

    def add_prod(self, lhs, rhs):
        if not lhs in self.prod.keys():
            self.nonterm.add(lhs)
            self.prod[lhs] = set()
        for part in rhs.split('|'):
            # TODO: автодобавление терминалов, из нового правила
            # self.term = self.term.union(set(part).difference(
            #     self.nonterm).difference(self.term))
            if len(part) > 0:
                self.prod[lhs].add(part)

    def simplify(self):
        self.remove_nonpoductive_prods(self.get_productive_prods())
        self.remove_unreachable_prods(self.get_reachable_prods())
        self.remove_emtpy_prods(self.get_empty_prods())
        self.resolve_chain_prods()
        self.remove_left_recursion()
        self.factorize_left()

    def remove_cycle_rules(self):
        # удаляем из правил типа A→A|...
        for key in self.prod.keys():
            if key in self.prod[key]:
                self.prod[key].remove(key)

    def remove_empty_terms(self):
        # удаляем из правил типа A→ε|...
        for production in self.prod.values():
            if 'ε' in production:
                production.remove('ε')

    def get_productive_prods(self):
        r0 = self.term
        while True:
            # заводим множество, чтобы после итерации проверить, добавились ли в него нетерминалы
            r1 = r0.copy()
            for key, production in self.prod.items():
                for p in production:
                    # если множество части p правил production является подмножеством терминалов V
                    if set(p).issubset(r0):
                        # то это значит, что нетерминал key порождает терминал(-ы) p непосредственно
                        # или key порождает нетерминал, который порождает нетерминал... который порождает терминал
                        r1.add(key)
            # если после итерации над правилом в множество r1 не был добавлен ни один нетерминал
            if len(r1) == len(r0):
                # возвращаем множество (терминалов U найденные нетерминалы ) без \ множества терминалов
                return r0.difference(self.term)
            else:
                r0 = r1

    def remove_nonpoductive_prods(self, productiveNonterms):
        # обновляем множество производящими нетерминалами
        self.nonterm = productiveNonterms
        d = copy.deepcopy(self.prod)
        for key, production in d.items():
            # если правило A→..., где A = key и A нетерминал, не в обновленном множестве нетерминалов
            if not key in self.nonterm:
                # удаляем это правило
                del self.prod[key]
            else:
                for p in production:
                    if len(set(p).difference(productiveNonterms.union(self.term))) != 0:
                        self.prod[key].remove(p)

    def get_reachable_prods(self):
        # главный нетерминал по определению является достижимым
        r0 = set(self.S)
        while True:
            # заводим множество, чтобы после итерации проверить, добавились ли в него нетерминалы
            r1 = r0.copy()
            # по каждому нетерминалу из множества порождающих нетерминалов
            for n in r0:
                # правую часть правила rhs
                rhs = set()
                for p in self.prod[n]:
                    rhs = rhs.union(set(p))
                # добавляем в множество с исключением терминалов
                r1 = r1.union(rhs.difference(self.term))
            # если после итерации над правилом в множество r1 не был добавлен ни один нетерминал
            if len(r1) == len(r0):
                break
            else:
                r0 = r1
        return r0

    def remove_unreachable_prods(self, reachableNonterms):
        # обновляем множество достижимыми нетерминалами
        self.nonterm = reachableNonterms
        d = self.prod.copy()
        for key in d.keys():
            # если правило A→..., где A = key и A нетерминал, не в обновленном множестве нетерминалов
            if not key in self.nonterm:
                # удаляем это правило
                del self.prod[key]

    def get_empty_prods(self):
        r0 = set('ε')
        while True:
            # заводим множество, чтобы после итерации проверить, добавились ли в него нетерминалы
            r1 = r0.copy()
            for lhs, rhs in self.prod.items():
                for part in rhs:
                    # если символ из правила это ε или нетерминал, порождающий ε, или нетерминал, порождающий
                    # этот нетерминал..., порождающий ε
                    if len(set(part).difference(r0)) == 0:
                        r1.add(lhs)
            # если после итерации над правилом в множество r1 не был добавлен ни один нетерминал
            if len(r1) == len(r0):
                # возвращаем множество (терминалов U найденные нетерминалы ) без \ множества терминалов
                return r0.difference(self.term)
            else:
                r0 = r1

    def remove_emtpy_prods(self, empty_nterms):
        # когда в S есть правило, которое может породить пустую цепочку
        # когда в S есть пустая цепочка
        # когда в S есть правило, содержащее S
        s_produces_e = False
        s_contains_e = False
        s_contains_s = False
        for part in self.prod[self.S]:
            if len(set(part).difference(empty_nterms)) == 0:
                s_produces_e = True
            if self.S in part:
                s_contains_s = True
                continue
            if 'ε' in part:
                s_contains_e = True
        # создаем полную копию правил, т.к. словарь по ходу момента нельзя менять
        d = copy.deepcopy(self.prod)
        for lhs, rhs in d.items():
            # если нетерминал key является таким, который порождает пустую цепочку
            if lhs in empty_nterms:
                for part in rhs:
                    # self.prod[lhs].remove(part)
                    for character in exclusions(part, empty_nterms):
                        self.prod[lhs].add(character)

                self.remove_cycle_rules()
                self.remove_empty_terms()

        if s_contains_e:
            if s_contains_s:
                tmp = self.S
                self.S = self.get_new_nonterm()
                self.add_prod(self.get_new_nonterm(), str(tmp + '|ε'))
            else:
                self.prod[self.S].add('ε')
        if s_produces_e:
            self.prod[self.S].add('ε')

    def resolve_chain_prods(self):
        # проходим по всем правилам
        for key in self.prod.keys():
            cycled_lhs = set()
            while True:
                # итерируем по копии правой части правила
                tmp = self.prod[key].copy()
                for part in tmp:
                    # если есть цепное правило
                    if len(part) == 1 and part[0] in self.nonterm and part not in cycled_lhs:
                        key2 = part[0]
                        cycled_lhs.add(key2)
                        # убираем его
                        self.prod[key].remove(key2)
                        # вместо него добавляем его правую часть
                        for p in self.prod[key2]:
                            if p in cycled_lhs or p == key:
                                continue
                            self.prod[key].add(p)
                # если по итогу не нашли цепные правила
                if len(self.prod[key] - tmp) == 0:
                    break
                # else:
                #     self.remove_cycle_rules()

    def remove_left_recursion(self):
        # проходим по копии ключей словаря (т.к. они добавятся в ходе работы)
        for key in set(self.prod.keys()):
            alpha = set()
            beta = set()
            for part in self.prod[key]:
                # если есть левая рекурсия
                if part[0] == key:
                    alpha.add(part[1:])
                else:
                    beta.add(part)
            if len(alpha) > 0:
                # костыльный вариант получения такой буквы, которая пока не встречалась в мн-ве нетерминалов
                newkey = self.get_new_nonterm()

                self.prod[key] = beta.copy()
                for b in beta:
                    self.prod[key].add(b + newkey)

                self.nonterm.add(newkey)
                self.prod[newkey] = alpha.copy()
                for a in alpha:
                    self.prod[newkey].add(a + newkey)

    def factorize_left(self):
        while True:
            flag = False
            for n in self.nonterm.copy():
                d = {}
                for i in range(1, len(max(self.prod[n], key=len))):
                    for part in self.prod[n]:
                        if len(part) >= i:
                            if d.get(part[0:i], False):
                                d[part[0:i]] += 1
                            else:
                                d[part[0:i]] = 1
                for part in self.prod[n]:
                    if d.get(part, False):
                        d.pop(part)
                if len(d) == 0:
                    continue
                prefix = (max(d, key=lambda key: d[key]))
                if d[prefix] > 1:
                    flag = True
                    newkey = self.get_new_nonterm()
                    self.nonterm.add(newkey)
                    self.prod[newkey] = set()
                    tmp = set()
                    tmp.add(prefix + newkey)
                    for part in self.prod[n]:
                        if prefix == part[0:len(prefix)] and len(part) > len(prefix):
                            self.prod[newkey].add(part[len(prefix):])
                            continue
                        tmp.add(part)
                    self.prod[n] = tmp
            if not flag:
                break

    def tocnf(self):
        newg = CFG(self.nonterm.copy(), self.term.copy(), {}, self.S)
        d = {}
        p0 = self.prod.copy()
        while True:
            iscnf = True
            newg.prod = {}
            for lhs, rhs in p0.items():
                for part in rhs:
                    if len(part) == 1:
                        newg.add_prod(lhs, part)
                        continue
                    if len(part) == 2:
                        # ??
                        if part[0] in newg.term:
                            # a?
                            if part[1] in newg.term:
                                # aa
                                newpart = ''
                                for p in part:
                                    if not p in d.keys():
                                        d[p] = self.get_new_nonterm(
                                            nonterm=newg.nonterm)
                                        newg.add_prod(d[p], p)
                                    newpart += d[p]
                                newg.add_prod(lhs, newpart)
                                continue
                            # aA
                            if not part[0] in d.keys():
                                d[part[0]] = self.get_new_nonterm(
                                    nonterm=newg.nonterm)
                            newg.add_prod(lhs, d[part[0]] + part[1])
                            newg.add_prod(d[part[0]], part[0])
                            continue
                        # A?
                        if part[1] in self.term:
                            # Aa
                            if not part[1] in d.keys():
                                d[part[1]] = self.get_new_nonterm(
                                    nonterm=newg.nonterm)
                            newg.add_prod(lhs, part[0] + d[part[1]])
                            newg.add_prod(d[part[1]], part[1])
                            continue
                        # AA
                        newg.add_prod(lhs, part)
                        continue
                    # a?..
                    iscnf = False
                    if part[0] in newg.term:
                        if not part[0] in d.keys():
                            d[part[0]] = self.get_new_nonterm(
                                nonterm=newg.nonterm)
                        newg.add_prod(lhs, d[part[0]] + part[1:])
                        newg.add_prod(d[part[0]], part[0])
                        continue
                    # A?..
                    newkey = self.get_new_nonterm(
                        nonterm=newg.nonterm)
                    newg.add_prod(lhs, part[0] + newkey)
                    newg.add_prod(newkey, part[1:])
            if iscnf:
                break
            p0 = newg.prod.copy()
        return newg

    def gen_random_convergent(self, lhs, cfactor=0.25, pcount={}):
        sentence = ''
        weights = []
        for prod in self.prod[lhs]:
            if prod in pcount:
                weights.append(cfactor ** (pcount[prod]))
            else:
                weights.append(1.0)

        rand_prod = self.prod[lhs][weighted_choice(weights)]
        pcount[rand_prod] += 1

        for sym in rand_prod:
            if sym in self.prod:
                sentence += self.gen_random_convergent(
                    sym,
                    cfactor=cfactor,
                    pcount=pcount)
            else:
                sentence += sym + ' '

        # backtracking: clear the modification to pcount
        pcount[rand_prod] -= 1
        return sentence

    def __str__(self):
        s = 'G: <N={'
        s += ', '.join(self.nonterm) + '}, '
        s += 'V={'
        s += ', '.join(self.term.difference('ε')) + '}, '
        s += 'P={'
        p = []
        for lhs, rhs in self.prod.items():
            tmp = ''
            tmp += lhs + '→'
            tmp += '|'.join(rhs)
            p.append(tmp)
        s += ', '.join(p)
        s += '}, S='
        s += self.S
        s += '>'
        return s


# cfg = CFG(['S'], ['k', 'l', 'm', 'n'], {
#     'S': 'kSl|kSm|n'
# }, 'S')
# cfg.factorize_left()
# cfg = CFG(['S'], ['a'], {
#     'S': 'aSa|a'
# }, 'S')

# cfg = CFG(['S', 'A', 'B', 'C'], ['a', 'b', 'c', ], {
#     'S': 'AaB|Aa|bc',
#     'A': 'AB|a|aC',
#     'B': 'Ba|b',
#     'C': 'AB|c'
# }, 'S')

# 6 S → aSa | a  | ε

# cfg7 = CFG(['S', 'Q'], ['a', 'b', 'c'], {
#     'S': 'aQb|accb',
#     'Q': 'cSc'
# }, 'S')
# cfg7.simplify()
# print('cfg7.prod = ', cfg7.prod)

# cfg8 = CFG(['S'], ['a', 'b'], {
#     'S': 'aSa|bSb|ε'
# }, 'S')
# cfg8.simplify()
# print('cfg8.prod = ', cfg8.prod)


# cfg9 = CFG(['S'], ['a', 'b'], {
#     'S': 'aSa|bSb|aa|bb'
# }, 'S')
# cfg9.simplify()
# print('cfg9.prod = ', cfg9.prod)


# cfg10 = CFG(['S', 'A', 'B'], ['a', 'b'], {
#     'S': 'aSB|bSA|aSBS|bSAS|ε',
#     'A': 'a',
#     'B': 'b'
# }, 'S')
# cfg10.simplify()
# print('cfg10.prod = ', cfg10.prod)

# cfg11 = CFG(['S'], ['a', 'b'], {
#     'S': 'aSb|bSa|SS|ε'
# }, 'S')
# cfg11.simplify()
# print('cfg11.prod = ', cfg11.prod)
#
# cfg12 = CFG(['S', 'A', 'B'], ['a', 'b'], {
#     'S': 'AB|aAb|ε|bBa'
# }, 'S')
# cfg12.simplify()
# print('cfg12.prod = ', cfg12.prod)
# cfg = CFG(['S', 'A', 'B', 'C'], ['a', 'b'], {
#     'S': 'ABC',
#     'A': 'BB|ε',
#     'B': 'CC|a',
#     'C': 'AA|b'
# }, 'S')
# cfg.remove_emtpy_prods(cfg.get_empty_prods())
# print('Удалили e-правила. P = ', cfg.prod)
# cfg.resolve_chain_prods()
# print('Удалили цепные правила. P = ', cfg.prod)
# cfg1 = CFG(p=['S→UX|VZ', 'T→aav|bb', 'U→aUa|bUb', 'V→aTb|bTa', 'W→YZY|aab', 'X→ε|Xa|Xb', 'Y→ε|YY|aU', 'Z→W|b'])
# cfg1.remove_nonpoductive_prods(cfg1.get_productive_prods())
# cfg1.remove_unreachable_prods(cfg1.get_reachable_prods())
# print(cfg1)
# cfg2 = CFG(p=['S → AB | CD', 'A → aB |  cE', 'B → c | cA | aB', 'C → dC', 'D → Dd | aA'])
# cfg2.remove_nonpoductive_prods(cfg2.get_productive_prods())
# cfg2.remove_unreachable_prods(cfg2.get_reachable_prods())
# print(cfg2)
# cfg3 = CFG(p=[
#      'S → BB |aBCD',
#     'B → ε| b ',
#     'C → ε| c ',
#     'D → d'
#
# ])
# cfg3.remove_nonpoductive_prods(cfg3.get_productive_prods())
# cfg3.remove_unreachable_prods(cfg3.get_reachable_prods())
# cfg3.remove_emtpy_prods(cfg3.get_empty_prods())
# print(cfg3)
# cfg4 = CFG(p=[
#      'C → ε | aSBb',
# 'S → ε | aSBb',
# 'A → ε | bbSA |  aaSS',
# 'B → ε | bbSA |  aaSS',
#
# ])
# cfg4.remove_nonpoductive_prods(cfg4.get_productive_prods())
# cfg4.remove_unreachable_prods(cfg4.get_reachable_prods())
# cfg4.remove_emtpy_prods(cfg4.get_empty_prods())
# print(cfg4)
# cfg5 = CFG(p=[
#      'S → BC | gDB',
#        'B → bCDE | ε',
#        'C → DaB | ac',
#        'D → Dd | ε',
#        'E → aSb | C'
#
# ])
# cfg5.remove_nonpoductive_prods(cfg5.get_productive_prods())
# cfg5.remove_unreachable_prods(cfg5.get_reachable_prods())
# cfg5.remove_emtpy_prods(cfg5.get_empty_prods())
# cfg5.resolve_chain_prods()
# print(cfg5)
# cfg6 = CFG(p=[
#      'S→AaB|aB|cC|Aa|a|c',
#      'A→AB|a|b|B',
#      'B→Ba|a',
#      'C→AB|A|B|c'
#
# ])
# cfg6.remove_nonpoductive_prods(cfg6.get_productive_prods())
# cfg6.remove_unreachable_prods(cfg6.get_reachable_prods())
# cfg6.remove_emtpy_prods(cfg6.get_empty_prods())
# cfg6.resolve_chain_prods()
# print(cfg6)
# cfg7 = CFG(p=[
#
# 'S → a | aC | B',
#       'B → B | bB | C',
#       'C → a | aS',
#
# ])
# cfg7.remove_nonpoductive_prods(cfg7.get_productive_prods())
# cfg7.remove_unreachable_prods(cfg7.get_reachable_prods())
# cfg7.remove_emtpy_prods(cfg7.get_empty_prods())
# cfg7.resolve_chain_prods()
# print(cfg7)
cfg8 = CFG(p=[
    'S → ASB | ε',
    'A → aAS | a',
    'B → SbS | A | bb'
])
cfg8.remove_nonpoductive_prods(cfg8.get_productive_prods())
cfg8.remove_unreachable_prods(cfg8.get_reachable_prods())
cfg8.remove_emtpy_prods(cfg8.get_empty_prods())
cfg8.resolve_chain_prods()
print(cfg8)
# cfg9 = CFG(p=[
# 'S → 0A0 | 1B1 | BB',
# 'A → C',
# 'B → S | A',
# 'C → S | ε'
# ], n=['S, A, B, C'], v=['0', '1'])
# cfg9.remove_nonpoductive_prods(cfg9.get_productive_prods())
# cfg9.remove_unreachable_prods(cfg9.get_reachable_prods())
# cfg9.remove_emtpy_prods(cfg9.get_empty_prods())
# cfg9.resolve_chain_prods()
# print(cfg9)
# cfg10 = CFG(p=['S → AAA | B',
# 'A → aA | B',
# 'B → ε'
#
# ], n=['S, A, B, C'], v=['0', '1'])
# cfg10.remove_nonpoductive_prods(cfg10.get_productive_prods())
# cfg10.remove_unreachable_prods(cfg10.get_reachable_prods())
# cfg10.remove_emtpy_prods(cfg10.get_empty_prods())
# cfg10.resolve_chain_prods()
# print(cfg10)
# cfg11 = CFG(p=[
# 'S → aAa | bBb | ε',
# 'A → C | a',
# 'B → C | b',
# 'C → CDE | ε',
# 'D → A | B | ab'
# ])
# cfg11.remove_nonpoductive_prods(cfg11.get_productive_prods())
# cfg11.remove_unreachable_prods(cfg11.get_reachable_prods())
# cfg11.remove_emtpy_prods(cfg11.get_empty_prods())
# cfg11.resolve_chain_prods()
# print(cfg11)
# cfg12 = CFG(p=[
# 'S → AB',
# 'A→ Aa|Ab|d|e',
# 'B → qK|rB|Bf |Bg',
# 'K → vS|w'
# ])
# cfg12.remove_nonpoductive_prods(cfg12.get_productive_prods())
# cfg12.remove_unreachable_prods(cfg12.get_reachable_prods())
# cfg12.remove_emtpy_prods(cfg12.get_empty_prods())
# cfg12.resolve_chain_prods()
# cfg12.remove_left_recursion()
# print(cfg12)
# cfg13 = CFG(p=[
# 'E → T|TR',
#     'R → +T|+TR',
#    'T → P|PF',
#    'F → *P|*PF',
#    'P → a|(E)',
#
#
# ], n=['E', 'T', 'R', 'F', 'P'], v=['+', '*', '(', ')', 'a'])
# cfg13.remove_nonpoductive_prods(cfg13.get_productive_prods())
# cfg13.remove_unreachable_prods(cfg13.get_reachable_prods())
# cfg13.remove_emtpy_prods(cfg13.get_empty_prods())
# cfg13.resolve_chain_prods()
# cfg13.remove_left_recursion()
# cfg13.factorize_left()
# print(cfg13)
