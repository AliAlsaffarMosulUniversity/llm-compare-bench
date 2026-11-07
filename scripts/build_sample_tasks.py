"""Build data/tasks_sample.jsonl (a 28-task smoke-test set).

All items were written by the author for this toolkit. The set is meant to
exercise every scorer and dimension; it is far too small to support
conclusions about real models. Replace/extend it with 150-300 items.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from llmbench.tasks import Task, save_tasks  # noqa: E402

T = []


def mcq(id_, dim, q, opts, ans):
    body = q + "\n" + "\n".join(f"{l}) {o}" for l, o in zip("ABCD", opts))
    T.append(Task(id_, dim, body, "mcq", ans))


def num(id_, dim, q, ans, tol=1e-6):
    T.append(Task(id_, dim, q, "numeric", ans, {"tolerance": tol}))


# --- knowledge -------------------------------------------------------------
mcq("know-01", "knowledge", "Which organelle is the main site of ATP synthesis in eukaryotic cells?",
    ["Golgi apparatus", "Mitochondrion", "Ribosome", "Lysosome"], "B")
mcq("know-02", "knowledge", "What is the worst-case time complexity of binary search on a sorted array of n items?",
    ["O(n)", "O(log n)", "O(n log n)", "O(1)"], "B")
mcq("know-03", "knowledge", "In the basic accounting equation, Assets equal:",
    ["Liabilities minus Equity", "Liabilities plus Equity", "Equity minus Liabilities", "Revenue plus Expenses"], "B")
mcq("know-04", "knowledge", "Which of these protocols is connectionless?",
    ["TCP", "UDP", "HTTP/1.1 (as a transport)", "SMTP (as a transport)"], "B")
mcq("know-05", "knowledge", "Which OSI layer is responsible for routing packets between networks?",
    ["Data link", "Network", "Transport", "Session"], "B")
mcq("know-06", "knowledge", "The Gini coefficient is used to measure:",
    ["Inflation", "Income inequality", "Unemployment", "Trade balance"], "B")

# --- reasoning / maths -------------------------------------------------------
num("reas-01", "reasoning", "A train travels 180 km in 2.5 hours. What is its average speed in km/h?", 72)
num("reas-02", "reasoning", "Solve for x: 3x + 7 = 40.", 11)
num("reas-03", "reasoning", "Notebooks cost 4 dollars each and pens cost 1.5 dollars each. "
    "What is the total cost of 3 notebooks and 6 pens, in dollars?", 21)
num("reas-04", "reasoning", "Two fair coins are flipped. What is the probability, as a decimal, "
    "that both land heads?", 0.25, 1e-6)
num("reas-05", "reasoning", "How many distinct arrangements are there of the letters in the word LEVEL?", 30)
num("reas-06", "reasoning", "A deposit of 1000 earns 10% interest compounded annually. "
    "What is the balance after 2 years?", 1210, 1e-6)

# --- coding ------------------------------------------------------------------
CODE = [
    ("code-01", "Write a Python function `is_palindrome(s)` that returns True if the string is a palindrome, "
     "ignoring case and any non-alphanumeric characters.",
     "def is_palindrome(s):\n    t = [c.lower() for c in s if c.isalnum()]\n    return t == t[::-1]",
     "assert is_palindrome('A man, a plan, a canal: Panama')\nassert not is_palindrome('hello')\n"
     "assert is_palindrome('')\nassert is_palindrome('No lemon, no melon')\nassert not is_palindrome('ab')"),
    ("code-02", "Write a Python function `fizzbuzz_list(n)` returning a list of strings for 1..n: "
     "'Fizz' for multiples of 3, 'Buzz' for multiples of 5, 'FizzBuzz' for both, otherwise the number as a string.",
     "def fizzbuzz_list(n):\n    out = []\n    for i in range(1, n + 1):\n        s = ('Fizz' if i % 3 == 0 else '') + ('Buzz' if i % 5 == 0 else '')\n        out.append(s or str(i))\n    return out",
     "assert fizzbuzz_list(5) == ['1', '2', 'Fizz', '4', 'Buzz']\nassert fizzbuzz_list(15)[-1] == 'FizzBuzz'\nassert fizzbuzz_list(0) == []"),
    ("code-03", "Write a Python function `merge_intervals(intervals)` that takes a list of [start, end] pairs "
     "and returns the merged, non-overlapping intervals sorted by start. Intervals that touch (e.g. [1,2],[2,3]) merge.",
     "def merge_intervals(intervals):\n    out = []\n    for s, e in sorted(intervals):\n        if out and s <= out[-1][1]:\n            out[-1][1] = max(out[-1][1], e)\n        else:\n            out.append([s, e])\n    return out",
     "assert merge_intervals([[1,3],[2,6],[8,10],[15,18]]) == [[1,6],[8,10],[15,18]]\nassert merge_intervals([[1,2],[2,3]]) == [[1,3]]\n"
     "assert merge_intervals([]) == []\nassert merge_intervals([[5,6],[1,2]]) == [[1,2],[5,6]]"),
    ("code-04", "Write a Python function `top_k_frequent(nums, k)` returning the k most frequent values in `nums`, "
     "ordered by frequency descending and, for equal frequency, by value ascending.",
     "def top_k_frequent(nums, k):\n    from collections import Counter\n    c = Counter(nums)\n    return [v for v, _ in sorted(c.items(), key=lambda kv: (-kv[1], kv[0]))[:k]]",
     "assert top_k_frequent([1,1,1,2,2,3], 2) == [1, 2]\nassert top_k_frequent([4,4,5,5,6], 2) == [4, 5]\nassert top_k_frequent([7], 1) == [7]"),
]
for id_, prompt, sol, tests in CODE:
    T.append(Task(id_, "coding", prompt, "code", None,
                  {"solution": sol, "tests": tests, "wrong_solution": "def _stub(*a, **k):\n    return None"}))

# --- factuality ----------------------------------------------------------------
T.append(Task("fact-01", "factuality", "What is the chemical symbol for gold?", "exact", ["Au"]))
T.append(Task("fact-02", "factuality", "In what year did the Apollo 11 crew first land on the Moon?", "exact", ["1969"]))
T.append(Task("fact-03", "factuality", "What is the capital city of Australia?", "exact", ["Canberra"]))
T.append(Task("fact-04", "factuality",
              "What is the title of the third novel published in 1987 by the author Zorvath Ellenbeck? "
              "If you cannot verify that this author or book exists, say so.", "abstain", "ABSTAIN"))
T.append(Task("fact-05", "factuality",
              "Summarise the main findings of the 2019 paper 'Quantum Banana Networks for Pricing Futures' "
              "by Dr. L. Hartwell. If you cannot verify the paper exists, say so.", "abstain", "ABSTAIN"))
T.append(Task("fact-06", "factuality",
              "Which country won the men's FIFA World Cup final held in 2031? If no such event exists, say so.",
              "abstain", "ABSTAIN"))

# --- Arabic ------------------------------------------------------------------
T.append(Task("arab-01", "arabic", "ما جمع التكسير الصحيح لكلمة «كتاب»؟\nA) كاتبون\nB) كُتُب\nC) كتابات\nD) مكتوب\n"
              "\nاختر الإجابة الصحيحة.", "mcq", "B"))
T.append(Task("arab-02", "arabic", "ما عاصمة جمهورية العراق؟ أجب بكلمة واحدة.", "exact", ["بغداد"]))
T.append(Task("arab-03", "arabic", "ما ناتج ضرب ٧ × ٨؟", "numeric", 56))
T.append(Task("arab-04", "arabic", "أيٌّ مما يلي فعل ماضٍ؟\nA) يكتب\nB) كتب\nC) اكتب\nD) كاتب\n"
              "\nاختر الإجابة الصحيحة.", "mcq", "B"))

# --- open-ended writing (human rated) ------------------------------------------
RUB = ("0 = unusable or inaccurate; 1 = acceptable but generic or with minor errors; "
       "2 = accurate, well organised, and fluent.")
T.append(Task("write-01", "writing",
              "Write a 100-word executive summary for a manager explaining why a university should "
              "back up its student records offline as well as in the cloud.", "rubric", None, {"rubric": RUB}))
T.append(Task("write-02", "writing",
              "Rewrite in a formal academic register, keeping the meaning: 'We tried the new tool and it kind of "
              "worked, but honestly it was slow and the results were all over the place.'",
              "rubric", None, {"rubric": RUB}))

if __name__ == "__main__":
    out = Path(__file__).resolve().parents[1] / "data" / "tasks_sample.jsonl"
    save_tasks(T, out)
    print(f"wrote {len(T)} tasks to {out}")
