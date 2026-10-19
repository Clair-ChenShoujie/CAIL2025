import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from cail.answers import error_type, normalize_answer, parse_answer, set_f1
from cail.data import normalize_row, prepare
from cail.evaluation import compare_runs, evaluate, submission
from cail.export import export_data
from cail.grpo import build_command
from cail.ifd import ifd_ratio, select_file
from cail.io import load_config, read_jsonl, write_jsonl
from cail.prompts import cot_valid, messages, qwen_prompt, teacher_messages
from cail.rewards import group_advantages, response_only, reward_breakdown
from cail.sft import pack_training_example


def question(number=1, answer="AB"):
    return {"id": f"q{number}", "statement": f"第{number}道法律题目", "option_list": {
        "A": "第一项内容", "B": "第二项内容", "C": "第三项内容", "D": "第四项内容"},
        "answer": answer, "subject": "民法", "type": "0"}


def cot(answer="AB"):
    return ("【法理分析】\n根据相关法律规则分析本题。\n"
            "【逐项排查】\nA. 第一项分析。B. 第二项分析。C. 第三项分析。D. 第四项分析。\n"
            f"【结论推导】\n综合上述分析得出结论。\n<answer>{answer}</answer>")


class AnswersTest(unittest.TestCase):
    def test_normalization(self):
        self.assertEqual(normalize_answer(["B", "A"]), "AB")
        self.assertEqual(normalize_answer("ｂ，ａ"), "AB")

    def test_invalid_answers(self):
        for value in ("", "AA", "ACE", None, "答案A"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                normalize_answer(value)

    def test_parser_does_not_read_option_mentions(self):
        self.assertIsNone(parse_answer("A 不成立，B 成立"))
        self.assertEqual(parse_answer("A 不成立。<answer>BD</answer>"), "BD")
        self.assertIsNone(parse_answer("<answer>A</answer><answer>B</answer>"))
        self.assertIsNone(parse_answer("<answer>AA</answer>"))

    def test_metrics_and_error_categories(self):
        self.assertAlmostEqual(set_f1("A", "AB"), 2 / 3)
        self.assertEqual(set_f1(None, "AB"), 0)
        for prediction, category in (("AB", "correct"), ("A", "missing_options"),
                                     ("ABC", "extra_options"), ("AC", "partial_overlap"),
                                     ("CD", "fully_wrong"), (None, "invalid_format")):
            self.assertEqual(error_type(prediction, "AB"), category)


class DataTest(unittest.TestCase):
    def test_schema(self):
        row = normalize_row(question())
        self.assertEqual(row["answer"], "AB")
        self.assertEqual(len(row["fingerprint"]), 64)
        for field, value in (("statement", None), ("option_list", [])):
            with self.subTest(field=field), self.assertRaises(ValueError):
                normalize_row({**question(), field: value})

    def test_prepare_is_reproducible_and_disjoint(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / "raw"
            first = [question(i) for i in range(20)]
            duplicate = {**first[0], "id": "duplicate"}
            conflicted = {**first[1], "id": "conflicting", "answer": "CD"}
            write_jsonl(source / "0_train.json", first + [duplicate, conflicted])
            write_jsonl(source / "1_train.json", [question(i, "C") for i in range(20, 40)])
            write_jsonl(source / "0_test.json", [question(2)])
            write_jsonl(source / "1_test.json", [question(100)])
            report = prepare(source, root / "one")
            other = prepare(source, root / "two")
            self.assertEqual(report, other)
            self.assertEqual(report["duplicates"], 1)
            self.assertEqual(report["conflicting_rows"], 2)
            self.assertEqual(report["test_overlap_removed"], 1)
            train = list(read_jsonl(root / "one/train.jsonl"))
            dev = list(read_jsonl(root / "one/dev.jsonl"))
            self.assertFalse({r["fingerprint"] for r in train} & {r["fingerprint"] for r in dev})
            self.assertEqual(len(train) + len(dev), 38)
            self.assertNotIn("answer", list(read_jsonl(root / "one/test.jsonl"))[0])

    def test_ifd_ratio_and_selection(self):
        self.assertEqual(ifd_ratio(0.5, 1.0), 0.5)
        for conditioned, unconditioned in ((0, 1), (1, 0), (math.nan, 1)):
            with self.assertRaises(ValueError):
                ifd_ratio(conditioned, unconditioned)
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            rows = [normalize_row(question(i)) for i in range(4)]
            scores = [{"id": row["id"], "fingerprint": row["fingerprint"], "ifd": score}
                      for row, score in zip(rows, (0.2, 0.9, 1.2, 0.8))]
            write_jsonl(root / "data.jsonl", rows)
            write_jsonl(root / "scores.jsonl", scores)
            select_file(root / "data.jsonl", root / "scores.jsonl", root / "out.jsonl", keep_count=2)
            self.assertEqual([r["id"] for r in read_jsonl(root / "out.jsonl")], ["q1", "q3"])

    def test_missing_ifd_score_is_detected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            write_jsonl(root / "data.jsonl", [normalize_row(question())])
            write_jsonl(root / "scores.jsonl", [])
            with self.assertRaises(ValueError):
                select_file(root / "data.jsonl", root / "scores.jsonl", root / "out.jsonl")


class PromptAndExportTest(unittest.TestCase):
    def test_teacher_has_answer_student_does_not(self):
        row = question()
        self.assertIn("标准答案：AB", teacher_messages(row)[-1]["content"])
        self.assertNotIn("标准答案", messages(row)[-1]["content"])
        self.assertTrue(qwen_prompt(row).endswith("<think>\n\n</think>\n\n"))

    def test_cot_validation(self):
        self.assertTrue(cot_valid(cot(), "AB"))
        self.assertFalse(cot_valid(cot(), "CD"))
        self.assertFalse(cot_valid(cot().replace("【法理分析】", ""), "AB"))
        self.assertFalse(cot_valid(cot() + "其他内容", "AB"))

    def test_export_keeps_label_outside_prompt(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            train, dev = normalize_row(question()), normalize_row(question(2))
            write_jsonl(root / "train.jsonl", [train])
            write_jsonl(root / "dev.jsonl", [dev])
            write_jsonl(root / "cot.jsonl", [{**train, "cot": cot()}])
            report = export_data(root / "cot.jsonl", root / "train.jsonl", root / "dev.jsonl", root / "out")
            self.assertEqual(report["sft"], 1)
            grpo = list(read_jsonl(root / "out/grpo_train.jsonl"))[0]
            self.assertEqual(grpo["label"], "AB")
            self.assertNotIn("标准答案", grpo["prompt"])
            self.assertIsInstance(grpo["prompt"], str)

    def test_export_rejects_leaked_dev_question(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            row = normalize_row(question())
            write_jsonl(root / "train.jsonl", [row])
            write_jsonl(root / "dev.jsonl", [{**row, "id": "same_question"}])
            write_jsonl(root / "cot.jsonl", [{**row, "cot": cot()}])
            with self.assertRaises(ValueError):
                export_data(root / "cot.jsonl", root / "train.jsonl", root / "dev.jsonl", root / "out")

    def test_sft_masks_prompt_and_keeps_answer(self):
        class Tokenizer:
            eos_token = "!"
            def apply_chat_template(self, items, **kwargs):
                return "PROMPT"
            def encode(self, text, **kwargs):
                return list(text.encode())
        items = messages(question()) + [{"role": "assistant", "content": "ANSWER"}]
        feature = pack_training_example(Tokenizer(), items, 100)
        self.assertEqual(feature["labels"][:6], [-100] * 6)
        self.assertEqual(feature["labels"][6:], list(b"ANSWER!"))
        self.assertIsNone(pack_training_example(Tokenizer(), items, 5))


class RewardTest(unittest.TestCase):
    def test_reward_orders_correct_partial_wrong(self):
        exact = reward_breakdown(cot("AB"), "AB")["reward"]
        partial = reward_breakdown(cot("A"), "AB")["reward"]
        wrong = reward_breakdown(cot("CD"), "AB")["reward"]
        self.assertAlmostEqual(exact, 1.0)
        self.assertGreater(exact, partial)
        self.assertGreater(partial, wrong)
        self.assertEqual(reward_breakdown("未作答", "AB")["reward"], -1)
        self.assertAlmostEqual(reward_breakdown(cot() + "<|im_end|>", "AB")["reward"], 1.0)

    def test_group_normalization(self):
        values = group_advantages([0.0, 1.0] * 8)
        self.assertAlmostEqual(sum(values), 0.0)
        self.assertEqual(group_advantages([1.0] * 8), [0.0] * 8)
        self.assertAlmostEqual(group_advantages([0.0, 1.0], 2)[1], 1 / math.sqrt(2), places=7)
        with self.assertRaises(ValueError):
            group_advantages([1.0] * 7)

    def test_response_isolated_from_prompt(self):
        prompt = "题目里的例子：<answer>A</answer>"
        self.assertEqual(response_only(prompt + cot(), prompt), cot())
        self.assertEqual(response_only("prefix<|im_start|>assistant\n" + cot(), "other"), cot())
        with self.assertRaises(ValueError):
            response_only("unknown", "prompt")

    def test_grpo_command_contract(self):
        command = build_command({}, "base", "prompts.jsonl", "out", "reward.py", "dev.jsonl", True)
        self.assertEqual(command[command.index("--n_samples_per_prompt") + 1], "8")
        self.assertEqual(command[command.index("--advantage_estimator") + 1], "group_norm")
        self.assertIn("--use_kl_loss", command)
        self.assertIn("--load_checkpoint", command)
        self.assertNotIn("--apply_chat_template", command)
        self.assertEqual(command[command.index("--label_key") + 1], "label")


class EvaluationTest(unittest.TestCase):
    def test_evaluation_metrics_and_bad_cases(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            write_jsonl(root / "gold.jsonl", [question(1), question(2), question(3, "C")])
            write_jsonl(root / "pred.jsonl", [{"id": "q1", "output": cot()},
                                              {"id": "q2", "output": cot("A")},
                                              {"id": "q3", "output": "A、B、C、D"}])
            result = evaluate(root / "gold.jsonl", root / "pred.jsonl", root / "eval")
            self.assertAlmostEqual(result["exact_accuracy"], 1 / 3)
            self.assertAlmostEqual(result["set_f1"], 5 / 9)
            self.assertAlmostEqual(result["micro_f1"], 0.75)
            self.assertAlmostEqual(result["partial_correct_among_errors"], 0.5)
            self.assertEqual(len(list(read_jsonl(root / "eval/bad_cases.jsonl"))), 2)
            self.assertEqual(result["by_answer_count"]["multiple"]["count"], 2)
            compare_runs([("test", root / "eval/metrics.json")], root / "compare.md")
            self.assertIn("test", (root / "compare.md").read_text(encoding="utf-8"))

    def test_missing_predictions(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            write_jsonl(root / "gold.jsonl", [question()])
            write_jsonl(root / "pred.jsonl", [])
            with self.assertRaises(ValueError):
                evaluate(root / "gold.jsonl", root / "pred.jsonl", root / "out")
            result = evaluate(root / "gold.jsonl", root / "pred.jsonl", root / "out", True)
            self.assertEqual(result["missing_predictions"], 1)
            self.assertEqual(result["exact_accuracy"], 0)

    def test_submission(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            write_jsonl(root / "pred.jsonl", [{"id": "q1", "output": cot()}])
            submission(root / "pred.jsonl", root / "answer.json")
            self.assertEqual(json.loads((root / "answer.json").read_text()), {"q1": ["A", "B"]})


class ConfigurationTest(unittest.TestCase):
    def test_inheritance(self):
        root = Path(__file__).resolve().parent.parent
        config = load_config(root / "configs/pipeline.linux.json")
        self.assertEqual(config["paths"]["model"], "/data/Qwen3-8B")
        self.assertEqual(config["cot"]["model"], "qwen-max")
        self.assertTrue(Path(config["paths"]["processed"]).is_absolute())

    def test_cli_help_needs_no_model_dependencies(self):
        from cail.cli import main
        with patch("sys.stdout"), self.assertRaises(SystemExit) as caught:
            main(["--help"])
        self.assertEqual(caught.exception.code, 0)


if __name__ == "__main__":
    unittest.main()
