import importlib.util
from pathlib import Path
import sys
import unittest


TOKENIZER_PATH = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "plugins"
    / "sta"
    / "tokenizer.py"
)
SPEC = importlib.util.spec_from_file_location("sta_tokenizer_for_test", TOKENIZER_PATH)
tokenizer = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = tokenizer
SPEC.loader.exec_module(tokenizer)


class StaTokenizerTest(unittest.TestCase):
    def analyze(self, texts, **kwargs):
        messages = [
            tokenizer.MessageSample(text=text, user_id=index % 5)
            for index, text in enumerate(texts)
        ]
        return tokenizer.analyze_messages(messages, **kwargs)

    def test_mixed_chinese_english_and_technical_tokens(self):
        analysis = self.analyze(
            [
                "用ChatGPT写Python代码，调用OpenAI API",
                "Node.js v2.1.0可以调用C++和C#",
            ],
            statistical_filter=False,
        )

        for word in {
            "ChatGPT", "Python", "OpenAI", "API", "Node.js", "v2.1.0",
            "C++", "C#", "代码", "调用",
        }:
            self.assertIn(word, analysis.frequencies)

    def test_casefold_merges_english_variants(self):
        analysis = self.analyze(
            ["API api Api"],
            statistical_filter=False,
        )

        self.assertEqual(1, len(analysis.frequencies))
        display = next(iter(analysis.frequencies))
        self.assertEqual(3, analysis.raw_counts[display])

    def test_learned_dictionary_assists_segmentation_but_stopword_wins(self):
        learned = self.analyze(
            ["赛博咕噜"],
            dictionary_words=["赛博咕噜"],
            statistical_filter=False,
        )
        stopped = self.analyze(
            ["赛博咕噜"],
            dictionary_words=["赛博咕噜"],
            stopwords=["赛博咕噜"],
            statistical_filter=False,
        )

        self.assertIn("赛博咕噜", learned.frequencies)
        self.assertNotIn("赛博咕噜", stopped.candidate_pool)

    def test_statistical_filter_keeps_group_term_and_repeated_pinyin(self):
        texts = ["感觉这个事情可以"] * 30
        texts.extend(["妈咪 cyk"] * 8)
        texts.append("qwertyasdf")

        analysis = self.analyze(texts, statistical_filter=True)

        self.assertIn("妈咪", analysis.frequencies)
        self.assertIn("cyk", analysis.frequencies)
        self.assertNotIn("qwertyasdf", analysis.frequencies)
        self.assertNotIn("感觉", analysis.frequencies)

    def test_high_frequency_group_term_is_not_treated_as_generic(self):
        analysis = self.analyze(
            ["妈咪"] * 25,
            statistical_filter=True,
        )

        self.assertIn("妈咪", analysis.frequencies)
        self.assertEqual(25, analysis.raw_counts["妈咪"])

    def test_generic_word_only_returns_on_a_daily_burst(self):
        history_days = []
        for day in range(10):
            history_days.append(
                [
                    tokenizer.MessageSample(
                        text="感觉 妈咪" if index < 2 else "妈咪",
                        user_id=(day + index) % 5,
                    )
                    for index in range(20)
                ]
            )
        baselines = tokenizer.build_generic_baselines(history_days)

        normal = self.analyze(
            ["感觉 妈咪"] * 3 + ["妈咪"] * 17,
            statistical_filter=True,
            generic_baselines=baselines,
        )
        burst = self.analyze(
            ["感觉 妈咪"] * 8 + ["妈咪"] * 12,
            statistical_filter=True,
            generic_baselines=baselines,
        )

        self.assertNotIn("感觉", normal.frequencies)
        self.assertIn("感觉", burst.frequencies)
        self.assertIn("感觉", burst.burst_words)

        protected_from_llm = tokenizer.apply_keyword_decisions(
            burst,
            [{"term": "感觉", "action": "drop", "canonical": "感觉"}],
        )
        self.assertIn("感觉", protected_from_llm.frequencies)

    def test_statistical_filter_switch_controls_singleton_pinyin(self):
        texts = ["普通消息"] * 24 + ["zxcvasdf"]

        filtered = self.analyze(texts, statistical_filter=True)
        unfiltered = self.analyze(texts, statistical_filter=False)

        self.assertNotIn("zxcvasdf", filtered.frequencies)
        self.assertIn("zxcvasdf", filtered.candidate_pool)
        self.assertFalse(filtered.candidate_pool["zxcvasdf"].locally_kept)
        self.assertIn("zxcvasdf", unfiltered.frequencies)

    def test_llm_can_restore_filtered_word_and_floor_refills_candidates(self):
        filtered = self.analyze(
            ["普通消息"] * 24 + ["awsl"],
            statistical_filter=True,
        )
        restored = tokenizer.apply_keyword_decisions(
            filtered,
            [{"term": "awsl", "action": "keep", "canonical": "awsl"}],
        )

        self.assertNotIn("awsl", filtered.frequencies)
        self.assertIn("awsl", restored.frequencies)

        wide_pool = self.analyze(
            ["普通消息"] * 20 + [f"term{index}" for index in range(12)],
            statistical_filter=True,
        )
        refilled = tokenizer.apply_keyword_decisions(
            wide_pool,
            [],
            min_keywords=10,
        )
        self.assertGreaterEqual(len(refilled.frequencies), 10)

        aliases = self.analyze(
            ["人工智能 AI"] * 2,
            statistical_filter=False,
        )
        merged = tokenizer.apply_keyword_decisions(
            aliases,
            [{"term": "人工智能", "action": "keep", "canonical": "AI"}],
            min_keywords=10,
        )
        self.assertNotIn("人工智能", merged.frequencies)

    def test_userword_overrides_generic_prior_but_stopword_still_wins(self):
        forced = self.analyze(
            ["感觉"],
            userwords=["感觉"],
            statistical_filter=True,
        )
        stopped = self.analyze(
            ["API api"],
            userwords=["API"],
            stopwords=["api"],
            statistical_filter=False,
        )

        self.assertIn("感觉", forced.frequencies)
        self.assertEqual({}, stopped.frequencies)

    def test_llm_decisions_cannot_drop_data_userword(self):
        analysis = self.analyze(
            ["妈咪 感觉", "妈咪 感觉"],
            userwords=["妈咪"],
            statistical_filter=False,
        )
        refined = tokenizer.apply_keyword_decisions(
            analysis,
            [
                {"term": "妈咪", "action": "drop", "canonical": "妈咪"},
                {"term": "感觉", "action": "drop", "canonical": "感觉"},
            ],
            protected_words=["妈咪"],
        )

        self.assertIn("妈咪", refined.frequencies)
        self.assertNotIn("感觉", refined.frequencies)

    def test_llm_payload_is_sanitized_and_output_is_candidate_bounded(self):
        import nonebot

        try:
            nonebot.get_driver()
        except ValueError:
            nonebot.init(host="127.0.0.1")
        nonebot.load_plugin("nonebot_plugin_apscheduler")
        from src.plugins.sta.llm_filter import (
            build_candidate_payload,
            collect_llm_dictionary_words,
            parse_llm_decisions,
            refine_keywords_with_llm,
        )
        import src.plugins.sta.llm_filter as llm_filter
        import src.plugins.sta.word_dictionary as word_dictionary

        messages = [
            tokenizer.MessageSample(
                "妈咪看这里 https://example.com 联系123456789",
                1,
            ),
            tokenizer.MessageSample("妈咪和cyk", 2),
        ]
        analysis = tokenizer.analyze_messages(
            messages,
            userwords=["妈咪"],
            statistical_filter=True,
        )
        analysis.burst_words = frozenset({"妈咪"})
        payload = build_candidate_payload(
            analysis,
            messages,
            protected_words=["妈咪"],
            candidate_limit=10,
            contexts_per_word=1,
        )
        mammy = next(item for item in payload["candidates"] if item["term"] == "妈咪")
        context = mammy["contexts"][0]

        self.assertTrue(mammy["protected"])
        self.assertTrue(mammy["daily_burst"])
        self.assertNotIn("example.com", context)
        self.assertNotIn("123456789", context)
        self.assertTrue(
            any(
                item["local_status"] == "filtered"
                for item in payload["candidates"]
            )
        )

        private_payload = build_candidate_payload(
            analysis,
            messages,
            protected_words=["妈咪"],
            candidate_limit=10,
            contexts_per_word=0,
        )
        self.assertTrue(
            all(not item["contexts"] for item in private_payload["candidates"])
        )

        terms = [item["term"] for item in payload["candidates"]]
        decisions = parse_llm_decisions(
            '{"decisions":['
            '{"term":"妈咪","action":"drop","canonical":"虚构词"},'
            '{"term":"虚构词","action":"keep","canonical":"虚构词"}'
            ']}',
            terms,
        )
        self.assertEqual(
            [{"term": "妈咪", "action": "drop", "canonical": "妈咪"}],
            decisions,
        )

        learned_words = collect_llm_dictionary_words(
            analysis,
            [{"term": "cyk", "action": "keep", "canonical": "cyk"}],
        )
        self.assertEqual(["cyk"], learned_words)

        class FakeDictionaryDB:
            def __init__(self):
                self.data = {}

            def get_copy(self, key, default=None):
                import copy
                return copy.deepcopy(self.data.get(key, default))

            def set(self, key, value):
                self.data[key] = value

        fake_db = FakeDictionaryDB()
        original_dictionary_db = word_dictionary.dictionary_db
        word_dictionary.dictionary_db = fake_db
        try:
            first_change = word_dictionary.record_llm_dictionary_words(
                ["cyk", "感觉", "blocked"],
                evidence_key="evidence-1",
                stopwords=["blocked"],
            )
            duplicate_change = word_dictionary.record_llm_dictionary_words(
                ["cyk"],
                evidence_key="evidence-1",
            )
            second_change = word_dictionary.record_llm_dictionary_words(
                ["cyk"],
                evidence_key="evidence-2",
            )
            stored_words = word_dictionary.get_llm_dictionary_words()
        finally:
            word_dictionary.dictionary_db = original_dictionary_db

        self.assertEqual(1, first_change)
        self.assertEqual(0, duplicate_change)
        self.assertEqual(1, second_change)
        self.assertEqual(["cyk"], stored_words)
        self.assertEqual(2, fake_db.data["words"]["cyk"]["confirmations"])

        class FailingChatSession:
            def __init__(self, *args, **kwargs):
                raise TimeoutError("test timeout")

        import asyncio
        original_session = llm_filter.ChatSession
        llm_filter.ChatSession = FailingChatSession
        try:
            fallback = asyncio.run(
                refine_keywords_with_llm(
                    group_id="sta-test-failure",
                    date_str="2099-01-01",
                    messages=messages,
                    analysis=analysis,
                    protected_words=["妈咪"],
                    model_name="test:model",
                    candidate_limit=10,
                    contexts_per_word=0,
                    min_keywords=2,
                )
            )
        finally:
            llm_filter.ChatSession = original_session
        self.assertGreaterEqual(len(fallback.frequencies), 2)


if __name__ == "__main__":
    unittest.main()
