"""tests.py  -  offline tests (no internet, no keys needed).  Run:  .venv\\Scripts\\python tests.py"""
import unittest
from unittest.mock import patch

import detector
import evidence


class FakeResp:
    def __init__(self, data, status=200):
        self._data, self.status_code = data, status

    def json(self):
        return self._data

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


def fake_get(quake=True, heavy_rain=False, rating="False", outlets=("A", "B")):
    """Builds a fake requests.get that answers like the 4 real APIs."""
    def _get(url, params=None, timeout=None):
        if "geocoding" in url:
            return FakeResp({"results": [{"name": "Lucknow", "country": "India", "latitude": 26.84, "longitude": 80.94}]})
        if "usgs" in url:
            f = [{"properties": {"mag": 5.1, "place": "near Lucknow", "url": "u"}}] if quake else []
            return FakeResp({"features": f})
        if "open-meteo" in url:
            return FakeResp({"daily": {"precipitation_sum": [2, 90 if heavy_rain else 5, 0]}})
        if "factchecktools" in url:
            return FakeResp({"claims": [{"text": "drinking hot water kills coronavirus", "claimReview": [
                {"textualRating": rating, "publisher": {"name": "PIB"}, "url": "x"}]}]})
        if "newsapi" in url:
            return FakeResp({"articles": [{"title": "t", "url": "u", "source": {"name": o}} for o in outlets]})
        raise AssertionError(url)
    return _get


class TestInputGate(unittest.TestCase):
    def test_personal(self):
        for t in ["I am unable to attend the class", "Sir I have fever so I cannot come to college tomorrow",
                  "Can you please send me the assignment", "How are you doing today"]:
            self.assertEqual(detector.check_input_type(t), "personal", t)

    def test_claims(self):
        for t in ["There is an emergency in Lucknow due to earthquake", "Drinking hot water kills coronavirus"]:
            self.assertEqual(detector.check_input_type(t), "claim", t)


class TestHelpers(unittest.TestCase):
    def test_event_and_place(self):
        t = "there is an emergency in lucknow due to earthquake"
        self.assertEqual(detector.detect_event(t), "earthquake")
        self.assertEqual(detector.place_candidates(t), ["lucknow"])
        self.assertEqual(detector.detect_event("Heavy rain in Delhi"), "rain")
        self.assertEqual(detector.place_candidates("flood in New Delhi today")[0], "New Delhi")

    def test_dateline_removed(self):
        self.assertNotIn("reuters", detector.clean_text("WASHINGTON (Reuters) - The senate passed a bill"))


class TestVerdict(unittest.TestCase):
    def ev(self, status):
        return {"source": "S", "status": status, "detail": "d", "url": ""}

    def test_rules(self):
        v = detector.make_verdict
        self.assertEqual(v("personal", [])["label"], "NOT NEWS")
        self.assertEqual(v("claim", [self.ev("contradicts")])["label"], "FAKE")
        self.assertEqual(v("claim", [self.ev("supports"), self.ev("no_data")])["label"], "REAL")
        self.assertEqual(v("claim", [self.ev("supports"), self.ev("contradicts")])["label"], "UNVERIFIED")
        self.assertEqual(v("claim", [self.ev("no_data"), self.ev("error"), self.ev("skipped")])["label"], "UNVERIFIED")


class TestPipelineWithFakeAPIs(unittest.TestCase):
    KEYS = {"NEWSAPI_KEY": "k", "GOOGLE_FACTCHECK_KEY": "k"}

    def run_analyze(self, text, **kw):
        with patch("evidence.requests.get", fake_get(**kw)):
            return evidence.analyze(text, self.KEYS)

    def test_personal_never_calls_api(self):
        with patch("evidence.requests.get", side_effect=AssertionError("API must not be called")):
            self.assertEqual(evidence.analyze("I am unable to attend the class", self.KEYS)["verdict"]["label"], "NOT NEWS")

    def test_earthquake_not_found_is_fake(self):
        r = self.run_analyze("There is an emergency in Lucknow due to earthquake", quake=False, outlets=())
        self.assertEqual(r["verdict"]["label"], "FAKE")
        self.assertEqual(r["evidence"][0]["status"], "contradicts")

    def test_earthquake_found_is_real(self):
        r = self.run_analyze("There is an emergency in Lucknow due to earthquake", quake=True, outlets=("A", "B"))
        self.assertEqual(r["verdict"]["label"], "REAL")

    def test_rain_closure_is_only_partial(self):
        r = self.run_analyze("All schools in Lucknow closed due to heavy rain", heavy_rain=True, outlets=())
        self.assertEqual(r["evidence"][0]["status"], "partial")
        self.assertEqual(r["verdict"]["label"], "UNVERIFIED")

    def test_factcheck_false_is_fake(self):
        r = self.run_analyze("Drinking hot water kills coronavirus", outlets=())
        self.assertEqual(r["verdict"]["label"], "FAKE")

    def test_api_down_is_unverified_not_fake(self):
        with patch("evidence.requests.get", side_effect=ConnectionError("no internet")):
            r = evidence.analyze("Drinking hot water kills coronavirus", self.KEYS)
        self.assertEqual(r["verdict"]["label"], "UNVERIFIED")

    def test_no_keys_is_unverified(self):
        with patch("evidence.requests.get", fake_get()):
            r = evidence.analyze("Drinking hot water kills coronavirus", {})
        self.assertEqual(r["verdict"]["label"], "UNVERIFIED")


class TestModel(unittest.TestCase):
    def test_short_text_skipped_and_long_text_scored(self):
        self.assertIsNone(detector.ml_style_score("Short statement"))
        long_text = ("The finance ministry said on Tuesday that the government will present the annual budget next month "
                     "and lawmakers from both parties are expected to debate the tax proposals in parliament "
                     "according to officials who spoke to reporters after the meeting on economic policy ") * 2
        out = detector.ml_style_score(long_text)
        self.assertIn(out["label"], ("REAL", "FAKE"))


if __name__ == "__main__":
    unittest.main(verbosity=1)
