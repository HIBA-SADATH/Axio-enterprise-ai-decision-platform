from django.test import TestCase


class CustomerSummaryTests(TestCase):
    def test_summary_returns_customer_metrics(self):
        response = self.client.get("/api/customers/summary/")

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertGreater(payload["total_customers"], 0)
        self.assertIn("segment_breakdown", payload)
        self.assertIn("region_breakdown", payload)
