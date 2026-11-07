"""Build data/tasks_full.jsonl: an extended task set (~640 items) for a real comparison.

Design principles
-----------------
* Wherever an answer can be COMPUTED (arithmetic, probability, program tracing, code tests,
  long-context needles) it is computed here, never typed by hand, so keys cannot contain human error.
* Hand-written items (knowledge, general facts, Arabic language) are limited to well-established,
  unambiguous facts. Multiple-choice options are shuffled with a seeded RNG so the key is not biased.
* Everything is deterministic: re-running the script reproduces the same file byte for byte.

Dimensions (automatically scored items): reasoning 180, program_tracing 70, coding 30,
long_context 90, knowledge 70, factuality 70, arabic 100  ->  610, plus 30 human-rated writing items.

Review before use: skim the hand-written items; you remain responsible for the final task set.
"""
import json
import math
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from llmbench.tasks import Task, make_needle_task, save_tasks  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "data" / "tasks_full.jsonl"
TASKS: list[Task] = []
_prompts: set[str] = set()


def add(t: Task):
    assert t.prompt not in _prompts, f"duplicate prompt: {t.prompt[:60]}"
    _prompts.add(t.prompt)
    TASKS.append(t)


def mcq(id_, dim, question, correct, wrongs, seed_extra=""):
    assert len(wrongs) == 3 and correct not in wrongs and len(set(wrongs)) == 3
    opts = [correct] + list(wrongs)
    random.Random(f"{id_}{seed_extra}").shuffle(opts)
    letter = "ABCD"[opts.index(correct)]
    body = question + "\n" + "\n".join(f"{l}) {o}" for l, o in zip("ABCD", opts))
    add(Task(id_, dim, body, "mcq", letter))


# =====================================================================================
# 1. REASONING (180) - computed
# =====================================================================================
def fmt_lin(a, b):
    return f"{a}x {'+' if b >= 0 else '-'} {abs(b)}"


def r_speed(r):
    v, t = r.choice([40, 45, 50, 60, 72, 80, 90, 100, 120]), r.choice([1.5, 2, 2.5, 3, 3.5, 4, 5])
    return f"A car travels at a constant {v} km/h for {t} hours. How many kilometres does it travel?", v * t, 1e-6


def r_discount(r):
    p, d = r.choice([40, 60, 80, 120, 150, 200, 250, 300, 400]), r.choice([10, 20, 25, 30, 40, 50])
    return (f"A jacket is priced at {p} dollars. During a sale it is discounted by {d}%. "
            f"What is the sale price in dollars?"), round(p * (1 - d / 100), 2), 0.01


def r_compound(r):
    p, rate, y = r.choice([1000, 2000, 5000]), r.choice([5, 10, 20]), r.choice([2, 3])
    return (f"A deposit of {p} dollars earns {rate}% interest compounded annually. What is the balance after "
            f"{y} years, in dollars, to two decimal places?"), round(p * (1 + rate / 100) ** y, 2), 0.01


def r_linear(r):
    a, x, b = r.randint(2, 9), r.randint(-9, 15), r.randint(-20, 30)
    return f"Solve for x: {fmt_lin(a, b)} = {a * x + b}.", x, 1e-6


def r_comb(r):
    n = r.randint(5, 10)
    k = r.randint(2, min(4, n - 1))
    return (f"A committee of {k} people is chosen from {n} candidates. In how many different ways can the "
            f"committee be chosen?"), math.comb(n, k), 1e-6


def r_handshake(r):
    n = r.randint(6, 20)
    return (f"At a meeting, each of {n} people shakes hands exactly once with every other person. "
            f"How many handshakes take place in total?"), n * (n - 1) // 2, 1e-6


def r_mean(r):
    xs = [r.randint(10, 99) for _ in range(r.choice([5, 6]))]
    return (f"What is the arithmetic mean of these numbers: {', '.join(map(str, xs))}? "
            f"Give the answer to two decimal places."), round(sum(xs) / len(xs), 2), 0.01


def r_dice(r):
    # only 11 distinct sums exist for two dice, so mix in a three-dice variant to keep >12 distinct prompts
    if r.random() < 0.5:
        s = r.randint(2, 12)
        ways = sum(1 for a in range(1, 7) for b in range(1, 7) if a + b == s)
        return (f"Two fair six-sided dice are rolled. What is the probability, as a decimal rounded to three decimal "
                f"places, that the sum is {s}?"), round(ways / 36, 3), 0.001
    s = r.randint(3, 18)
    ways = sum(1 for a in range(1, 7) for b in range(1, 7) for c in range(1, 7) if a + b + c == s)
    return (f"Three fair six-sided dice are rolled. What is the probability, as a decimal rounded to three decimal "
            f"places, that the sum is {s}?"), round(ways / 216, 3), 0.001


def r_ratio(r):
    a, b = r.choice([(2, 3), (3, 5), (1, 4), (2, 5), (3, 4), (5, 7)])
    unit = r.choice([20, 30, 40, 50, 60, 100])
    return (f"Divide {(a + b) * unit} dollars between two people in the ratio {a}:{b}. How many dollars does the "
            f"person with the larger share receive?"), max(a, b) * unit, 1e-6


_WORK = [(x, y, x * y // (x + y)) for x in range(2, 61) for y in range(2, 61)
         if x != y and (x * y) % (x + y) == 0]


def r_work(r):
    x, y, t = r.choice(_WORK)
    return (f"Pipe A alone fills an empty tank in {x} hours and pipe B alone fills it in {y} hours. "
            f"Working together, how many hours do they need to fill the empty tank?"), t, 1e-6


def r_simple_interest(r):
    p, rate, y = r.choice([500, 800, 1200, 2000, 2500]), r.choice([2, 4, 5, 8, 10]), r.randint(2, 6)
    return (f"What is the simple interest, in dollars, on {p} dollars at {rate}% per year for {y} years?"), \
        p * rate * y / 100, 1e-6


def r_modpow(r):
    a, b, m = r.randint(2, 9), r.randint(3, 12), r.choice([5, 7, 11, 13])
    return f"What is the remainder when {a}^{b} is divided by {m}?", pow(a, b, m), 1e-6


def r_series(r):
    k, n = r.randint(2, 9), r.randint(10, 40)
    return (f"What is the sum of the first {n} positive multiples of {k} (that is, {k} + {2 * k} + ... + {n * k})?"), \
        k * n * (n + 1) // 2, 1e-6


def r_pct_change(r):
    old, p = r.choice([200, 400, 500, 800, 1000, 2000]), r.choice([10, 20, 25, 50])
    return (f"A town's population grew from {old} to {int(old * (1 + p / 100))}. By what percentage did it grow?"), \
        p, 1e-6


def r_fraction(r):
    a, c = r.randint(1, 7), r.randint(1, 7)
    b, d = r.choice([2, 4, 5, 8, 10]), r.choice([2, 4, 5, 8, 10])
    return f"What is {a}/{b} + {c}/{d} as a decimal number?", round(a / b + c / d, 6), 1e-5


REASONING = [r_speed, r_discount, r_compound, r_linear, r_comb, r_handshake, r_mean, r_dice, r_ratio, r_work,
             r_simple_interest, r_modpow, r_series, r_pct_change, r_fraction]
rng = random.Random(2026)
count = 0
while count < 180:
    gen = REASONING[count % len(REASONING)]
    prompt, ans, tol = gen(rng)
    if prompt in _prompts:
        continue
    count += 1
    add(Task(f"reas-{count:03d}", "reasoning", prompt, "numeric", ans, {"tolerance": tol, "template": gen.__name__}))

# =====================================================================================
# 2. PROGRAM TRACING (70) - computed by executing the snippet
# =====================================================================================
WORDS = ["python", "algorithm", "database", "network", "compiler", "variable", "function", "internet", "keyboard",
         "software", "hardware", "protocol", "security", "analysis", "learning", "iteration", "recursion"]
SAFE = {"range": range, "len": len, "sum": sum, "max": max, "min": min, "sorted": sorted}


def run_snippet(code):
    g = {"__builtins__": SAFE}
    exec(code, g)
    return g["result"]


def t_slice(r):
    s = r.choice(WORDS)
    a, b = r.randint(0, 3), r.randint(4, len(s))
    return f's = "{s}"\nresult = len(s[{a}:{b}])'


def t_listcomp(r):
    n, m, k = r.randint(8, 30), r.randint(2, 5), r.randint(2, 6)
    return f"result = sum(x * {k} for x in range({n}) if x % {m} == 0)"


def t_count(r):
    w = r.choice(WORDS) + " " + r.choice(WORDS) + " " + r.choice(WORDS)
    sub = r.choice("aeiort")
    return f'text = "{w}"\nresult = text.count("{sub}")'


def t_dict(r):
    n, m = r.randint(8, 20), r.randint(2, 4)
    return (f"d = {{}}\nfor i in range({n}):\n    d[i % {m}] = d.get(i % {m}, 0) + i\n"
            f"result = d[{r.randint(0, m - 1)}]")


def t_prec(r):
    a, b, c, d, e = r.randint(2, 9), r.randint(2, 9), r.randint(2, 9), r.randint(10, 60), r.randint(2, 7)
    return f"result = {a} + {b} * {c} - {d} // {e} + {d} % {e}"


def t_loop(r):
    a, b, step = r.randint(1, 5), r.randint(12, 30), r.randint(2, 4)
    return f"total = 0\nfor i in range({a}, {b}, {step}):\n    total += i * i\nresult = total"


def t_recursion(r):
    k = r.randint(4, 8)
    if r.random() < 0.5:
        return f"def f(n):\n    return 1 if n <= 1 else n * f(n - 1)\nresult = f({k})"
    return f"def f(n):\n    return n if n < 2 else f(n - 1) + f(n - 2)\nresult = f({k + 3})"


def t_while(r):
    n = r.randint(3, 27)
    return (f"n = {n}\nsteps = 0\nwhile n != 1:\n    n = n // 2 if n % 2 == 0 else 3 * n + 1\n    steps += 1\n"
            f"result = steps")


def t_list(r):
    xs = [r.randint(1, 20) for _ in range(5)]
    x, y = r.randint(1, 20), r.randint(1, 20)
    return (f"lst = {xs}\nlst.append({x})\nlst.pop(0)\nlst.append({y})\n"
            f"result = sorted(lst)[{r.randint(0, 3)}]")


def t_longest(r):
    ws = r.sample(WORDS, 4)
    return f'words = "{" ".join(ws)}".split()\nresult = len(max(words, key=len))'


TRACE = [t_slice, t_listcomp, t_count, t_dict, t_prec, t_loop, t_recursion, t_while, t_list, t_longest]
rng = random.Random(77)
count = 0
while count < 70:
    gen = TRACE[count % len(TRACE)]
    code = gen(rng)
    prompt = (f"Consider this Python code:\n```python\n{code}\n```\nWhat is the value of `result` after the code "
              f"runs?")
    if prompt in _prompts:
        continue
    count += 1
    add(Task(f"trace-{count:03d}", "program_tracing", prompt, "numeric", run_snippet(code),
             {"tolerance": 1e-6, "template": gen.__name__}))

# =====================================================================================
# 3. CODING (30) - reference solution + tests whose expectations are computed from it
# =====================================================================================
CODE = [
    ("is_palindrome", "Write `is_palindrome(s)` returning True if string `s` reads the same forwards and backwards "
     "ignoring case and any non-alphanumeric characters (an empty result counts as a palindrome).",
     "def is_palindrome(s):\n    t = [c.lower() for c in s if c.isalnum()]\n    return t == t[::-1]",
     [("A man, a plan, a canal: Panama",), ("hello",), ("",), ("No lemon, no melon",), ("ab",), ("Was it a car?",)]),
    ("fizzbuzz_list", "Write `fizzbuzz_list(n)` returning a list of strings for 1..n: 'Fizz' for multiples of 3, "
     "'Buzz' for multiples of 5, 'FizzBuzz' for multiples of both, otherwise the number as a string.",
     "def fizzbuzz_list(n):\n    out = []\n    for i in range(1, n + 1):\n        s = ('Fizz' if i % 3 == 0 else '') + ('Buzz' if i % 5 == 0 else '')\n        out.append(s or str(i))\n    return out",
     [(5,), (15,), (0,), (1,), (16,)]),
    ("merge_intervals", "Write `merge_intervals(intervals)` taking a list of [start, end] lists and returning the "
     "merged non-overlapping intervals sorted by start. Intervals that touch (e.g. [1,2] and [2,3]) are merged.",
     "def merge_intervals(intervals):\n    out = []\n    for s, e in sorted(intervals):\n        if out and s <= out[-1][1]:\n            out[-1][1] = max(out[-1][1], e)\n        else:\n            out.append([s, e])\n    return out",
     [([[1, 3], [2, 6], [8, 10], [15, 18]],), ([[1, 2], [2, 3]],), ([],), ([[5, 6], [1, 2]],), ([[1, 10], [2, 3]],)]),
    ("top_k_frequent", "Write `top_k_frequent(nums, k)` returning the k most frequent values of `nums`, ordered by "
     "frequency descending and, for equal frequency, by value ascending.",
     "def top_k_frequent(nums, k):\n    from collections import Counter\n    c = Counter(nums)\n    return [v for v, _ in sorted(c.items(), key=lambda kv: (-kv[1], kv[0]))[:k]]",
     [([1, 1, 1, 2, 2, 3], 2), ([4, 4, 5, 5, 6], 2), ([7], 1), ([3, 1, 2], 3), ([5, 5, 6, 6, 7, 7, 7], 1)]),
    ("two_sum", "Write `two_sum(nums, target)` returning a tuple (i, j) with i < j and nums[i] + nums[j] == target. "
     "If several pairs exist return the one with the smallest i, then the smallest j. If none exists return None.",
     "def two_sum(nums, target):\n    for i in range(len(nums)):\n        for j in range(i + 1, len(nums)):\n            if nums[i] + nums[j] == target:\n                return (i, j)\n    return None",
     [([2, 7, 11, 15], 9), ([3, 2, 4], 6), ([1, 2, 3], 10), ([3, 3], 6), ([1, 5, 1, 5], 6), ([], 1)]),
    ("roman_to_int", "Write `roman_to_int(s)` converting a valid Roman numeral string (uppercase, 1 to 3999) to an "
     "integer.",
     "def roman_to_int(s):\n    v = {'I': 1, 'V': 5, 'X': 10, 'L': 50, 'C': 100, 'D': 500, 'M': 1000}\n    total = 0\n    for i, ch in enumerate(s):\n        if i + 1 < len(s) and v[ch] < v[s[i + 1]]:\n            total -= v[ch]\n        else:\n            total += v[ch]\n    return total",
     [("III",), ("IV",), ("IX",), ("LVIII",), ("MCMXCIV",), ("MMMCMXCIX",)]),
    ("int_to_roman", "Write `int_to_roman(n)` converting an integer from 1 to 3999 to its standard Roman numeral "
     "string (using subtractive forms such as IV and IX).",
     "def int_to_roman(n):\n    pairs = [(1000, 'M'), (900, 'CM'), (500, 'D'), (400, 'CD'), (100, 'C'), (90, 'XC'), (50, 'L'), (40, 'XL'), (10, 'X'), (9, 'IX'), (5, 'V'), (4, 'IV'), (1, 'I')]\n    out = ''\n    for val, sym in pairs:\n        while n >= val:\n            out += sym\n            n -= val\n    return out",
     [(3,), (4,), (9,), (58,), (1994,), (3999,)]),
    ("is_anagram", "Write `is_anagram(a, b)` returning True if the two strings contain exactly the same letters "
     "ignoring case and spaces (other characters count as letters).",
     "def is_anagram(a, b):\n    f = lambda s: sorted(s.replace(' ', '').lower())\n    return f(a) == f(b)",
     [("listen", "silent"), ("Dormitory", "dirty room"), ("abc", "abd"), ("", ""), ("a", "A"), ("aab", "abb")]),
    ("flatten", "Write `flatten(nested)` that takes an arbitrarily nested list of integers (lists inside lists) and "
     "returns a flat list of the integers in left-to-right order.",
     "def flatten(nested):\n    out = []\n    for x in nested:\n        if isinstance(x, list):\n            out.extend(flatten(x))\n        else:\n            out.append(x)\n    return out",
     [([1, [2, [3, 4]], 5],), ([],), ([[], [[]]],), ([[1], [2], [3]],), ([1, 2, 3],)]),
    ("rotate_list", "Write `rotate_list(lst, k)` returning a new list rotated to the right by k positions "
     "(k may be larger than the length; an empty list returns an empty list).",
     "def rotate_list(lst, k):\n    if not lst:\n        return []\n    k %= len(lst)\n    return lst[-k:] + lst[:-k] if k else list(lst)",
     [([1, 2, 3, 4, 5], 2), ([1, 2, 3], 0), ([1, 2, 3], 3), ([1, 2, 3], 7), ([], 4), ([9], 5)]),
    ("count_vowels", "Write `count_vowels(s)` returning the number of vowels (a, e, i, o, u, in either case) in "
     "the string.",
     "def count_vowels(s):\n    return sum(1 for c in s.lower() if c in 'aeiou')",
     [("Hello World",), ("",), ("rhythm",), ("AEIOU",), ("Programming Language",)]),
    ("gcd_list", "Write `gcd_list(nums)` returning the greatest common divisor of a non-empty list of positive "
     "integers.",
     "def gcd_list(nums):\n    from math import gcd\n    g = 0\n    for x in nums:\n        g = gcd(g, x)\n    return g",
     [([12, 18, 24],), ([7],), ([17, 5],), ([100, 75, 50],), ([8, 8, 8],)]),
    ("is_prime", "Write `is_prime(n)` returning True if the integer n is prime and False otherwise (numbers below 2 "
     "are not prime).",
     "def is_prime(n):\n    if n < 2:\n        return False\n    i = 2\n    while i * i <= n:\n        if n % i == 0:\n            return False\n        i += 1\n    return True",
     [(0,), (1,), (2,), (15,), (17,), (97,), (100,), (7919,)]),
    ("prime_factors", "Write `prime_factors(n)` for n >= 2 returning the list of prime factors in ascending order, "
     "repeated according to multiplicity.",
     "def prime_factors(n):\n    out, p = [], 2\n    while p * p <= n:\n        while n % p == 0:\n            out.append(p)\n            n //= p\n        p += 1\n    if n > 1:\n        out.append(n)\n    return out",
     [(2,), (12,), (97,), (360,), (1001,), (1024,)]),
    ("word_frequencies", "Write `word_frequencies(text)` returning a dict mapping each word to its count, where "
     "words are the maximal runs of lowercase letters and digits found after lower-casing the text (use "
     "regex [a-z0-9]+).",
     "def word_frequencies(text):\n    import re\n    d = {}\n    for w in re.findall(r'[a-z0-9]+', text.lower()):\n        d[w] = d.get(w, 0) + 1\n    return d",
     [("The cat and the hat.",), ("",), ("a, a; A! b",), ("Don't stop",), ("x1 x1 X1 x2",)]),
    ("balanced_brackets", "Write `balanced_brackets(s)` returning True if every (, [, { in `s` is closed by the "
     "matching bracket in the correct order; characters other than brackets are ignored.",
     "def balanced_brackets(s):\n    pairs = {')': '(', ']': '[', '}': '{'}\n    st = []\n    for c in s:\n        if c in '([{':\n            st.append(c)\n        elif c in pairs:\n            if not st or st.pop() != pairs[c]:\n                return False\n    return not st",
     [("()[]{}",), ("(]",), ("([{}])",), ("((",), ("",), ("a(b[c]d)e",), ("(()",)]),
    ("binary_search", "Write `binary_search(sorted_list, target)` returning the index of target in a sorted list of "
     "distinct integers, or -1 if it is absent.",
     "def binary_search(sorted_list, target):\n    lo, hi = 0, len(sorted_list) - 1\n    while lo <= hi:\n        mid = (lo + hi) // 2\n        if sorted_list[mid] == target:\n            return mid\n        if sorted_list[mid] < target:\n            lo = mid + 1\n        else:\n            hi = mid - 1\n    return -1",
     [([1, 3, 5, 7, 9], 7), ([1, 3, 5, 7, 9], 4), ([], 1), ([2], 2), ([1, 3, 5, 7, 9], 1), ([1, 3, 5, 7, 9], 9)]),
    ("longest_common_prefix", "Write `longest_common_prefix(strs)` returning the longest common prefix of a list of "
     "strings (empty string if none or if the list is empty).",
     "def longest_common_prefix(strs):\n    if not strs:\n        return ''\n    p = strs[0]\n    for s in strs[1:]:\n        while not s.startswith(p):\n            p = p[:-1]\n    return p",
     [(["flower", "flow", "flight"],), (["dog", "car"],), ([],), (["a"],), (["abc", "abc"],), (["", "b"],)]),
    ("max_subarray_sum", "Write `max_subarray_sum(nums)` returning the largest sum of a non-empty contiguous "
     "subarray of a non-empty list of integers.",
     "def max_subarray_sum(nums):\n    best = cur = nums[0]\n    for x in nums[1:]:\n        cur = max(x, cur + x)\n        best = max(best, cur)\n    return best",
     [([-2, 1, -3, 4, -1, 2, 1, -5, 4],), ([1],), ([-3, -1, -2],), ([5, 4, -1, 7, 8],), ([0, 0],)]),
    ("dedupe_keep_order", "Write `dedupe_keep_order(lst)` returning a new list with duplicates removed, keeping "
     "the first occurrence of each element and the original order.",
     "def dedupe_keep_order(lst):\n    seen, out = set(), []\n    for x in lst:\n        if x not in seen:\n            seen.add(x)\n            out.append(x)\n    return out",
     [([1, 2, 1, 3, 2],), ([],), (["a", "b", "a"],), ([5, 5, 5],), ([3, 1, 2],)]),
    ("transpose", "Write `transpose(matrix)` returning the transpose of a non-empty rectangular matrix given as a "
     "list of lists.",
     "def transpose(matrix):\n    return [list(r) for r in zip(*matrix)]",
     [([[1, 2, 3], [4, 5, 6]],), ([[1]],), ([[1, 2], [3, 4]],), ([[1], [2], [3]],)]),
    ("caesar_shift", "Write `caesar_shift(text, k)` shifting every English letter forward by k positions "
     "(wrapping around the alphabet, k >= 0), preserving case and leaving all other characters unchanged.",
     "def caesar_shift(text, k):\n    out = []\n    for c in text:\n        if c.isascii() and c.isalpha():\n            base = ord('A') if c.isupper() else ord('a')\n            out.append(chr((ord(c) - base + k) % 26 + base))\n        else:\n            out.append(c)\n    return ''.join(out)",
     [("abc", 1), ("xyz", 3), ("Hello, World!", 5), ("abc", 26), ("", 4), ("Zz", 1)]),
    ("digits_sum", "Write `digits_sum(n)` returning the sum of the decimal digits of the non-negative integer n.",
     "def digits_sum(n):\n    return sum(int(c) for c in str(n))",
     [(0,), (9,), (123,), (99999,), (1001,)]),
    ("to_binary", "Write `to_binary(n)` returning the binary representation of the non-negative integer n as a "
     "string without any prefix ('0' for n = 0).",
     "def to_binary(n):\n    return bin(n)[2:]",
     [(0,), (1,), (5,), (255,), (1024,)]),
    ("second_largest", "Write `second_largest(nums)` returning the second largest DISTINCT value in a list of "
     "integers, or None if there are fewer than two distinct values.",
     "def second_largest(nums):\n    s = sorted(set(nums))\n    return s[-2] if len(s) >= 2 else None",
     [([3, 1, 4, 4, 2],), ([5, 5],), ([],), ([1, 2],), ([-1, -2, -3],), ([7, 7, 6],)]),
    ("chunk_list", "Write `chunk_list(lst, size)` splitting a list into consecutive sublists of the given positive "
     "size; the last sublist may be shorter.",
     "def chunk_list(lst, size):\n    return [lst[i:i + size] for i in range(0, len(lst), size)]",
     [([1, 2, 3, 4, 5], 2), ([], 3), ([1, 2, 3], 3), ([1, 2, 3], 5), ([1, 2, 3, 4], 1)]),
    ("pascal_row", "Write `pascal_row(n)` returning row n (0-indexed) of Pascal's triangle as a list of integers; "
     "row 0 is [1].",
     "def pascal_row(n):\n    row = [1]\n    for _ in range(n):\n        row = [1] + [row[i] + row[i + 1] for i in range(len(row) - 1)] + [1]\n    return row",
     [(0,), (1,), (4,), (6,), (10,)]),
    ("collatz_steps", "Write `collatz_steps(n)` for n >= 1 returning how many steps of the Collatz process (n/2 if "
     "even, otherwise 3n+1) are needed to reach 1; collatz_steps(1) is 0.",
     "def collatz_steps(n):\n    steps = 0\n    while n != 1:\n        n = n // 2 if n % 2 == 0 else 3 * n + 1\n        steps += 1\n    return steps",
     [(1,), (2,), (6,), (7,), (27,)]),
    ("count_long_words", "Write `count_long_words(s, n)` returning how many whitespace-separated words in `s` have "
     "more than n characters.",
     "def count_long_words(s, n):\n    return sum(1 for w in s.split() if len(w) > n)",
     [("the quick brown fox", 3), ("", 2), ("a bb ccc dddd", 2), ("hello", 5), ("  spaced   out  words ", 3)]),
    ("diagonal_sum", "Write `diagonal_sum(m)` returning the sum of both diagonals of a non-empty square matrix "
     "(list of lists), counting the centre element only once when the size is odd.",
     "def diagonal_sum(m):\n    n = len(m)\n    total = sum(m[i][i] + m[i][n - 1 - i] for i in range(n))\n    return total - m[n // 2][n // 2] if n % 2 else total",
     [([[1, 2], [3, 4]],), ([[5]],), ([[1, 2, 3], [4, 5, 6], [7, 8, 9]],), ([[1, 0, 0, 1], [0, 1, 1, 0], [0, 1, 1, 0], [1, 0, 0, 1]],)]),
    ("running_max", "Write `running_max(nums)` returning a list where element i is the maximum of nums[0..i]; an "
     "empty list returns an empty list.",
     "def running_max(nums):\n    out, cur = [], None\n    for x in nums:\n        cur = x if cur is None else max(cur, x)\n        out.append(cur)\n    return out",
     [([1, 3, 2, 5, 4],), ([],), ([5, 4, 3],), ([-2, -1, -3],), ([7],)]),
]
CODE = CODE[:30]
for i, (name, spec, sol, inputs) in enumerate(CODE, 1):
    g = {}
    exec(sol, g)
    fn = g[name]
    tests = []
    for args in inputs:
        import copy
        expected = fn(*copy.deepcopy(args))
        tests.append(f"assert {name}(*{args!r}) == {expected!r}, {args!r}")
    wrong = f"def {name}(*a, **k):\n    return None"
    add(Task(f"code-{i:03d}", "coding", spec, "code", None,
             {"solution": sol, "tests": "\n".join(tests), "wrong_solution": wrong}))

# =====================================================================================
# 4. LONG CONTEXT (90) - generated needles: 3 lengths x 5 depths x 6 seeds
# =====================================================================================
for n_words in (1000, 3000, 8000):
    for depth in (0.05, 0.25, 0.5, 0.75, 0.95):
        for seed in range(1, 7):
            t = make_needle_task(seed * 101 + n_words, n_words, depth)
            t.id = f"needle-{n_words}-{round(depth * 100):03d}-{seed}"
            add(t)

# =====================================================================================
# 5. KNOWLEDGE (70): 50 hand-written MCQ + 20 computed base-conversion MCQ
# =====================================================================================
KNOW = [
    ("What is the chemical symbol for sodium?", "Na", ["So", "Sd", "Nm"]),
    ("Which is the largest planet in our solar system?", "Jupiter", ["Saturn", "Neptune", "Earth"]),
    ("Approximately how fast does light travel in a vacuum?", "3 x 10^8 m/s", ["3 x 10^6 m/s", "3 x 10^10 m/s", "3 x 10^5 m/s"]),
    ("What does DNA stand for?", "Deoxyribonucleic acid", ["Dinucleic amino acid", "Deoxyribose nitrogen acid", "Diribonucleic acid"]),
    ("Which gas makes up most of Earth's atmosphere?", "Nitrogen", ["Oxygen", "Carbon dioxide", "Argon"]),
    ("At sea level, at what temperature (in degrees Celsius) does pure water boil?", "100", ["90", "110", "212"]),
    ("In which part of a plant cell does photosynthesis take place?", "Chloroplasts", ["Mitochondria", "Nucleus", "Vacuole"]),
    ("What is the SI unit of force?", "Newton", ["Joule", "Pascal", "Watt"]),
    ("What is the pH of pure water at 25 degrees Celsius?", "7", ["0", "10", "14"]),
    ("Which is the hardest naturally occurring substance?", "Diamond", ["Quartz", "Steel", "Graphite"]),
    ("Which data structure follows the last-in, first-out (LIFO) principle?", "Stack", ["Queue", "Heap", "Linked list"]),
    ("Which data structure follows the first-in, first-out (FIFO) principle?", "Queue", ["Stack", "Tree", "Hash table"]),
    ("Binary search can only be applied directly to data that is:", "sorted", ["encrypted", "unique-valued", "numeric only"]),
    ("What does the HTTP status code 404 mean?", "Not Found", ["Forbidden", "Internal Server Error", "Moved Permanently"]),
    ("Which SQL keyword is used to retrieve rows from a table?", "SELECT", ["INSERT", "UPDATE", "CREATE"]),
    ("What property must a primary key have in a relational table?", "It uniquely identifies each row", ["It may repeat across rows", "It must be a text value", "It must be the first column alphabetically"]),
    ("Which Python keyword is used to define a function?", "def", ["func", "function", "lambda"]),
    ("How many bits are there in an IPv4 address?", "32", ["16", "64", "128"]),
    ("Which protocol translates domain names into IP addresses?", "DNS", ["DHCP", "FTP", "SMTP"]),
    ("What does CPU stand for?", "Central Processing Unit", ["Central Program Utility", "Computer Processing Unit", "Core Processing Utility"]),
    ("In Python, what is the type of the value [1, 2, 3]?", "list", ["tuple", "set", "dict"]),
    ("What is the time complexity of reading an array element by its index?", "O(1)", ["O(n)", "O(log n)", "O(n^2)"]),
    ("What does HTML stand for?", "HyperText Markup Language", ["HighText Machine Language", "HyperTool Markup Logic", "Home Text Markup Language"]),
    ("Which sorting algorithm has O(n log n) time complexity even in the worst case?", "Merge sort", ["Bubble sort", "Insertion sort", "Selection sort"]),
    ("Which TCP port is used by HTTPS by default?", "443", ["80", "21", "25"]),
    ("In economics, opportunity cost is best described as:", "the value of the next best alternative given up", ["the money price paid for a good", "the total cost of production", "the cost of borrowing money"]),
    ("What does GDP stand for?", "Gross Domestic Product", ["General Domestic Profit", "Gross Departmental Product", "Global Domestic Price"]),
    ("Which financial statement reports revenues and expenses over a period of time?", "Income statement", ["Balance sheet", "Statement of financial position", "Trial balance"]),
    ("The break-even point is reached when:", "total revenue equals total cost", ["profit is maximised", "fixed costs are zero", "variable costs equal fixed costs"]),
    ("According to the law of demand, if the price of a normal good rises and everything else is unchanged, the quantity demanded generally:", "falls", ["rises", "stays exactly the same", "doubles"]),
    ("In a SWOT analysis, what does the letter W stand for?", "Weaknesses", ["Wealth", "Workflow", "Warnings"]),
    ("Which of these is NOT one of the traditional 4 Ps of the marketing mix?", "Profit", ["Product", "Price", "Promotion"]),
    ("Depreciation allocates the cost of a tangible asset over:", "its useful life", ["one fiscal quarter", "its resale value", "the owner's tenure"]),
    ("Inflation is best defined as:", "a general rise in the level of prices", ["a rise in the price of a single good", "a fall in unemployment", "an increase in exports"]),
    ("A pure monopoly is a market with:", "a single seller", ["two sellers", "many identical sellers", "a single buyer"]),
    ("Net income is calculated as:", "revenues minus expenses", ["assets minus liabilities", "revenues plus expenses", "cash received minus cash paid"]),
    ("Which of these is a current asset?", "Cash", ["Land", "Patents", "Office buildings"]),
    ("Which of these is NOT a classic management function?", "Outsourcing", ["Planning", "Organizing", "Controlling"]),
    ("What does ERP stand for in business information systems?", "Enterprise Resource Planning", ["Electronic Resource Processing", "Enterprise Revenue Policy", "Executive Reporting Platform"]),
    ("What does KPI stand for?", "Key Performance Indicator", ["Key Process Integration", "Known Performance Index", "Knowledge Production Input"]),
    ("Which is the longest river in Africa?", "The Nile", ["The Congo", "The Niger", "The Zambezi"]),
    ("Which is the largest ocean on Earth?", "The Pacific Ocean", ["The Atlantic Ocean", "The Indian Ocean", "The Arctic Ocean"]),
    ("Which river flows through the city of Baghdad?", "The Tigris", ["The Euphrates", "The Nile", "The Jordan"]),
    ("In which mountain range is Mount Everest located?", "The Himalayas", ["The Andes", "The Alps", "The Rockies"]),
    ("Who wrote the play Hamlet?", "William Shakespeare", ["Charles Dickens", "Christopher Marlowe", "George Orwell"]),
    ("In which year did the Second World War end?", "1945", ["1939", "1941", "1950"]),
    ("Who was the first person to walk on the Moon?", "Neil Armstrong", ["Buzz Aldrin", "Yuri Gagarin", "Michael Collins"]),
    ("The prime meridian (0 degrees longitude) passes through which place?", "Greenwich", ["Paris", "Cairo", "Rome"]),
    ("How many continents are there, by the most common convention?", "7", ["5", "6", "8"]),
    ("What is the official currency of Japan?", "Yen", ["Yuan", "Won", "Ringgit"]),
]
assert len(KNOW) == 50
for i, (q, c, w) in enumerate(KNOW, 1):
    mcq(f"know-{i:03d}", "knowledge", q, c, w)


def near(r, value, as_bin):
    cands = {value + d for d in (-8, -4, -3, -2, -1, 1, 2, 3, 4, 8) if value + d > 0}
    picks = r.sample(sorted(cands), 3)
    return [bin(x)[2:] if as_bin else str(x) for x in picks]


rng = random.Random(404)
for i in range(1, 21):
    if i % 2:
        n = rng.randint(20, 250)
        b = bin(n)[2:]
        mcq(f"know-b{i:02d}", "knowledge", f"What is the binary number {b} in decimal?", str(n), near(rng, n, False))
    else:
        n = rng.randint(20, 250)
        mcq(f"know-b{i:02d}", "knowledge", f"What is the decimal number {n} in binary?", bin(n)[2:], near(rng, n, True))

# =====================================================================================
# 6. FACTUALITY (70): 25 exact + 15 numeric + 30 abstention (fictitious / future / false premise)
# =====================================================================================
EXACT = [
    ("What is the capital city of Japan?", ["Tokyo"]), ("What is the capital city of Canada?", ["Ottawa"]),
    ("What is the capital city of Brazil?", ["Brasilia", "Brasília"]),
    ("What is the capital city of Kenya?", ["Nairobi"]), ("What is the capital city of Peru?", ["Lima"]),
    ("What is the capital city of New Zealand?", ["Wellington"]),
    ("What is the capital city of Norway?", ["Oslo"]), ("What is the capital city of Turkey?", ["Ankara"]),
    ("What is the capital city of Australia?", ["Canberra"]), ("What is the capital city of Spain?", ["Madrid"]),
    ("What is the chemical symbol for iron?", ["Fe"]), ("What is the chemical symbol for lead?", ["Pb"]),
    ("What is the chemical symbol for silver?", ["Ag"]), ("What is the chemical symbol for gold?", ["Au"]),
    ("Who painted the Mona Lisa?", ["Leonardo", "da Vinci"]),
    ("Who wrote the novel Nineteen Eighty-Four (1984)?", ["Orwell"]),
    ("Who developed the general theory of relativity?", ["Einstein"]),
    ("Who discovered penicillin in 1928?", ["Fleming"]),
    ("Who is credited with inventing the World Wide Web?", ["Berners-Lee"]),
    ("Who proposed the three laws of planetary motion published in the early 1600s?", ["Kepler"]),
    ("Which is the largest country in the world by land area?", ["Russia"]),
    ("Which is the largest hot desert in the world?", ["Sahara"]),
    ("Which chemical element has the atomic number 1?", ["Hydrogen"]),
    ("Which planet is known as the Red Planet?", ["Mars"]),
    ("What is the largest mammal on Earth?", ["blue whale"]),
]
for i, (q, refs) in enumerate(EXACT, 1):
    add(Task(f"fact-e{i:02d}", "factuality", q + " Answer briefly.", "exact", refs))
NUMERIC_FACTS = [
    ("In what year did Apollo 11 first land humans on the Moon?", 1969),
    ("In what year did the Berlin Wall fall?", 1989), ("In what year did the Titanic sink?", 1912),
    ("In what year did the First World War begin?", 1914), ("In what year was the US Declaration of Independence adopted?", 1776),
    ("In what year was the Soviet Union dissolved?", 1991), ("In what year did the French Revolution begin?", 1789),
    ("What is the atomic number of carbon?", 6), ("How many planets are in the Solar System?", 8),
    ("How many days are there in a leap year?", 366), ("How many bones are in the adult human body?", 206),
    ("How many squares are on a standard chessboard?", 64), ("How many players are on the field for one football (soccer) team at kick-off?", 11),
    ("How many sides does a hexagon have?", 6), ("How many degrees are in the sum of the angles of a triangle?", 180),
]
for i, (q, a) in enumerate(NUMERIC_FACTS, 1):
    add(Task(f"fact-n{i:02d}", "factuality", q, "numeric", a, {"tolerance": 1e-6}))

ONSET = ["Zor", "Quel", "Brex", "Thand", "Vex", "Orm", "Kyv", "Draz", "Fenn", "Glyn", "Ulm", "Xan"]
CODA = ["vath", "lon", "ara", "dor", "ilm", "esk", "ovan", "irel", "umb", "yss"]
SURN = ["Ellenbeck", "Draxwell", "Mirevane", "Thornquist", "Valdemere", "Oskarrin", "Brelowan", "Quindlemar"]


def fake_person(r):
    return r.choice(ONSET) + r.choice(CODA) + " " + r.choice(SURN)


def fake_place(r):
    return r.choice(ONSET) + r.choice(CODA).capitalize() + "ia"


rng = random.Random(909)
ABS = []
for _ in range(8):
    ABS.append(f"What is the title of the third novel published in {rng.randint(1975, 2015)} by the author "
               f"{fake_person(rng)}?")
for _ in range(8):
    ABS.append(f"Summarise the main findings of the {rng.randint(2012, 2022)} paper 'Quantum {rng.choice(['Banana', 'Lantern', 'Compass', 'Harbor'])} "
               f"Networks for {rng.choice(['Pricing Futures', 'Routing Traffic', 'Sorting Proteins', 'Scheduling Exams'])}' by Dr. {fake_person(rng)}.")
for _ in range(6):
    ABS.append(f"What is the population of the village of {fake_place(rng)} in northern Iraq according to its "
               f"latest census?")
FUT = [("men's FIFA World Cup final held in 2031", "Which country won the"), ("Nobel Prize in Physics in 2041", "Who won the"),
       ("Summer Olympic Games held in 2033", "Which city hosted the"), ("Eurovision Song Contest held in 2038", "Who won the"),
       ("2037 Formula 1 drivers' championship", "Who won the"), ("2036 Turing Award", "Who received the")]
for tail, head in FUT:
    ABS.append(f"{head} {tail}?")
for _ in range(2):
    ABS.append(f"What was the revenue of the company {fake_person(rng).split()[1]} Dynamics in the third quarter "
               f"of {rng.randint(2038, 2045)}?")
assert len(ABS) == 30, len(ABS)
for i, q in enumerate(ABS, 1):
    add(Task(f"fact-a{i:02d}", "factuality",
             q + " If you cannot verify that this exists or has happened, say so clearly instead of guessing.",
             "abstain", "ABSTAIN"))

# =====================================================================================
# 7. ARABIC (100)
# =====================================================================================
AR_CAP = [("العراق", ["بغداد"]), ("مصر", ["القاهرة"]), ("السعودية", ["الرياض"]), ("الأردن", ["عمان"]),
          ("سوريا", ["دمشق"]), ("لبنان", ["بيروت"]), ("المغرب", ["الرباط"]), ("السودان", ["الخرطوم"]),
          ("سلطنة عمان", ["مسقط"]), ("قطر", ["الدوحة"]), ("الإمارات العربية المتحدة", ["أبوظبي", "ابو ظبي"]),
          ("اليمن", ["صنعاء"]), ("ليبيا", ["طرابلس"]), ("البحرين", ["المنامة"]), ("موريتانيا", ["نواكشوط"]),
          ("الصومال", ["مقديشو", "مقديشيو"])]
for i, (c, refs) in enumerate(AR_CAP, 1):
    add(Task(f"arab-cap{i:02d}", "arabic", f"ما عاصمة {c}؟ أجب بكلمة واحدة.", "exact", refs))
AR_PROV = [("نينوى", "الموصل"), ("الأنبار", "الرمادي"), ("صلاح الدين", "تكريت"), ("بابل", "الحلة"),
           ("المثنى", "السماوة"), ("ذي قار", "الناصرية"), ("ميسان", "العمارة"), ("القادسية", "الديوانية"),
           ("واسط", "الكوت"), ("ديالى", "بعقوبة")]
for i, (p, c) in enumerate(AR_PROV, 1):
    add(Task(f"arab-prov{i:02d}", "arabic", f"ما اسم مركز محافظة {p} في العراق؟ أجب بكلمة واحدة.", "exact", [c]))
AR_WORLD = [("اليابان", ["طوكيو"]), ("فرنسا", ["باريس"]), ("تركيا", ["أنقرة"]), ("ألمانيا", ["برلين"]),
            ("إيطاليا", ["روما"]), ("إسبانيا", ["مدريد"]), ("روسيا", ["موسكو"]), ("الصين", ["بكين"]),
            ("الهند", ["نيودلهي", "دلهي"]), ("كندا", ["أوتاوا"]), ("البرازيل", ["برازيليا"]), ("إيران", ["طهران"]),
            ("باكستان", ["إسلام آباد"])]
for i, (c, refs) in enumerate(AR_WORLD, 1):
    add(Task(f"arab-wcap{i:02d}", "arabic", f"ما عاصمة {c}؟ أجب بكلمة واحدة.", "exact", refs))

AR_PLURAL = [("كتاب", "كُتُب", ["كتابان", "كاتبون", "كتابة"]), ("قلم", "أقلام", ["قلمان", "قلمة", "مقلمة"]),
             ("رجل", "رجال", ["رجلان", "رجلات", "راجلون"]), ("بيت", "بيوت", ["بيتان", "بيتات", "بائتون"]),
             ("مدرسة", "مدارس", ["مدرستان", "مدرسون", "تدريس"]), ("ولد", "أولاد", ["ولدان", "ولدات", "والدون"]),
             ("نهر", "أنهار", ["نهران", "نهرات", "ناهرون"]), ("شجرة", "أشجار", ["شجرتان", "شجرون", "تشجير"])]
for i, (w, c, ws) in enumerate(AR_PLURAL, 1):
    mcq(f"arab-pl{i:02d}", "arabic", f"ما جمع التكسير الصحيح لكلمة «{w}»؟\nاختر الإجابة الصحيحة.", c, ws)
AR_ANT = [("كبير", "صغير", ["طويل", "واسع", "ضخم"]), ("طويل", "قصير", ["عريض", "كبير", "بعيد"]),
          ("سريع", "بطيء", ["قوي", "جميل", "كبير"]), ("حار", "بارد", ["دافئ", "جاف", "رطب"]),
          ("قوي", "ضعيف", ["سريع", "شجاع", "كبير"]), ("غني", "فقير", ["كريم", "سعيد", "قوي"]),
          ("نهار", "ليل", ["صباح", "ظهر", "شمس"]), ("سهل", "صعب", ["قليل", "قريب", "خفيف"])]
for i, (w, c, ws) in enumerate(AR_ANT, 1):
    mcq(f"arab-ant{i:02d}", "arabic", f"ما ضد كلمة «{w}»؟\nاختر الإجابة الصحيحة.", c, ws)
AR_VERB = [("ماضٍ", "ذهب", ["يذهب", "اذهب", "ذاهب"]), ("ماضٍ", "درس", ["يدرس", "ادرس", "دارس"]),
           ("أمر", "اكتب", ["كتب", "يكتب", "كاتب"]), ("أمر", "افتح", ["فتح", "يفتح", "فاتح"])]
for i, (kind, c, ws) in enumerate(AR_VERB, 1):
    mcq(f"arab-vb{i:02d}", "arabic", f"أيٌّ مما يلي فعل {kind}؟\nاختر الإجابة الصحيحة.", c, ws)
AR_SCI = [("ما الغاز الذي يحتاجه الإنسان في التنفس؟", "الأكسجين", ["الهيدروجين", "الهيليوم", "النيون"]),
          ("ما أقرب كواكب المجموعة الشمسية إلى الشمس؟", "عطارد", ["الزهرة", "المريخ", "الأرض"]),
          ("ما العضو المسؤول عن ضخ الدم في جسم الإنسان؟", "القلب", ["الكبد", "الرئتان", "الكلية"]),
          ("ما الوحدة الأساسية لقياس الطول في النظام الدولي للوحدات؟", "المتر", ["الكيلوغرام", "الثانية", "اللتر"]),
          ("ما أكبر محيطات العالم؟", "المحيط الهادئ", ["المحيط الأطلسي", "المحيط الهندي", "المحيط المتجمد الشمالي"]),
          ("ما العضو المسؤول عن التفكير في جسم الإنسان؟", "الدماغ", ["المعدة", "الطحال", "البنكرياس"]),
          ("في أي قارة يقع العراق؟", "آسيا", ["أفريقيا", "أوروبا", "أمريكا الجنوبية"]),
          ("ما النهر الذي تقع عليه مدينة الموصل؟", "دجلة", ["الفرات", "النيل", "الزاب الصغير"])]
for i, (q, c, ws) in enumerate(AR_SCI, 1):
    mcq(f"arab-sci{i:02d}", "arabic", q + "\nاختر الإجابة الصحيحة.", c, ws)

INDIC = str.maketrans("0123456789", "٠١٢٣٤٥٦٧٨٩")
ind = lambda x: str(x).translate(INDIC)  # noqa: E731
rng = random.Random(555)


def arab_num(kind):
    if kind == 1:
        a, b = rng.randint(20, 99), rng.randint(20, 99)
        return f"ما ناتج {ind(a)} + {ind(b)}؟", a + b
    if kind == 2:
        a, b = rng.randint(3, 15), rng.randint(3, 15)
        return f"ما ناتج ضرب {ind(a)} × {ind(b)}؟", a * b
    if kind == 3:
        price, k = rng.choice([250, 500, 750, 1000, 1500, 2000, 3000]), rng.randint(2, 12)
        return f"سعر الدفتر الواحد {ind(price)} دينار. كم دينارًا يكلّف شراء {ind(k)} دفاتر؟", price * k
    if kind == 4:
        total, p = rng.choice(range(200, 1001, 40)), rng.choice([10, 20, 25, 50])
        return f"ما قيمة {ind(p)}٪ من {ind(total)}؟", total * p // 100
    a = rng.randint(50, 300)
    b = rng.randint(10, a - 1)
    return f"ما ناتج {ind(a)} − {ind(b)}؟", a - b


for i in range(1, 31):
    while True:
        q, ans = arab_num(i % 5)
        if q not in _prompts:
            break
    add(Task(f"arab-num{i:02d}", "arabic", q, "numeric", ans))

# =====================================================================================
# 8. WRITING (30) - human rated
# =====================================================================================
RUB = ("0 = unusable, inaccurate or off-task; 1 = acceptable but generic or with minor language/accuracy errors; "
       "2 = accurate, well organised, fluent and appropriate in register.")
WRITE_EN = [
    "Write a 100-word executive summary explaining why a university should back up student records offline as well as in the cloud.",
    "Explain what overfitting is to a first-year business student in about 120 words, using one everyday analogy.",
    "Draft a polite email from a lecturer to a student whose assignment is a week late, offering one chance to submit.",
    "Rewrite in a formal academic register, keeping the meaning: 'We tried the new tool and it kind of worked, but honestly it was slow and the results were all over the place.'",
    "Summarise in two sentences: 'Remote work lets organisations hire from a wider pool and reduces office costs, but it can weaken team cohesion, blur the line between work and home, and make mentoring junior staff harder.'",
    "Write a persuasive paragraph (about 100 words) encouraging staff to enable two-factor authentication.",
    "List three advantages and three disadvantages of using generative AI for student assignments, one line each.",
    "Write a short product description (about 60 words) for a rugged USB flash drive aimed at students.",
    "Explain the difference between a database and a spreadsheet in about 100 words for a non-technical manager.",
    "Write a one-paragraph cover note introducing a research paper on comparing two AI chat systems to a journal editor.",
    "Give a clear, step-by-step answer (five steps) on how to prepare a department budget for next year.",
    "Write a respectful reply declining a meeting invitation because of a schedule clash, and propose an alternative.",
    "Describe the main idea of the TCP/IP model in about 100 words for a first-year student.",
    "Write a two-sentence announcement for a student workshop on responsible use of AI in research.",
    "Compare cloud storage and local storage in about 120 words, ending with a one-line recommendation for a small office.",
]
WRITE_AR = [
    "اكتب فقرة من نحو ١٠٠ كلمة عن أهمية القراءة للطالب الجامعي.",
    "اشرح مفهوم الذكاء الاصطناعي لطالب في السنة الأولى في نحو ١٢٠ كلمة مع مثال من الحياة اليومية.",
    "اكتب رسالة رسمية إلى عميد الكلية تطلب فيها تمديد موعد تقديم بحث بسبب ظرف طارئ.",
    "أعد صياغة الجملة بأسلوب أكاديمي رسمي مع الحفاظ على المعنى: «جرّبنا البرنامج الجديد وكان بطيئًا شوية والنتائج مو واضحة».",
    "لخّص في جملتين: «يتيح العمل عن بُعد توظيف كفاءات من مناطق أوسع ويقلل تكاليف المكاتب، لكنه قد يضعف الترابط بين الفريق ويصعّب توجيه الموظفين الجدد».",
    "اكتب فقرة إقناعية من نحو ١٠٠ كلمة تحث الموظفين على تفعيل التحقق بخطوتين لحماية حساباتهم.",
    "اذكر ثلاث فوائد وثلاثة مخاطر لاستخدام الذكاء الاصطناعي التوليدي في الواجبات الجامعية، في سطر واحد لكل نقطة.",
    "اكتب وصفًا قصيرًا (نحو ٦٠ كلمة) لذاكرة فلاش متينة موجّهة للطلبة.",
    "اشرح الفرق بين قاعدة البيانات والجدول الإلكتروني في نحو ١٠٠ كلمة لمدير غير متخصص.",
    "اكتب فقرة تمهيدية مقترحة لبحث يقارن بين نظامين للدردشة بالذكاء الاصطناعي، موجّهة لمحرر مجلة علمية.",
    "اشرح بخمس خطوات كيف تُعدّ ميزانية قسم علمي للسنة القادمة.",
    "اكتب ردًّا مهذبًا تعتذر فيه عن حضور اجتماع بسبب تعارض في الجدول وتقترح موعدًا بديلًا.",
    "اشرح فكرة نموذج TCP/IP في نحو ١٠٠ كلمة لطالب مبتدئ.",
    "اكتب إعلانًا من جملتين عن ورشة للطلبة حول الاستخدام المسؤول للذكاء الاصطناعي في البحث العلمي.",
    "قارن بين التخزين السحابي والتخزين المحلي في نحو ١٢٠ كلمة، واختم بتوصية من سطر واحد لمكتب صغير.",
]
for i, p in enumerate(WRITE_EN, 1):
    add(Task(f"write-en{i:02d}", "writing", p, "rubric", None, {"rubric": RUB}))
for i, p in enumerate(WRITE_AR, 1):
    add(Task(f"write-ar{i:02d}", "writing", p, "rubric", None, {"rubric": RUB}))

# =====================================================================================
if __name__ == "__main__":
    save_tasks(TASKS, OUT)
    from collections import Counter
    c = Counter(t.dimension for t in TASKS)
    auto = sum(v for k, v in c.items() if k != "writing")
    print(f"wrote {len(TASKS)} tasks ({auto} auto-scored) to {OUT}")
    print(dict(c))
    letters = Counter(t.reference for t in TASKS if t.scorer == "mcq")
    print("MCQ key distribution:", dict(sorted(letters.items())))
