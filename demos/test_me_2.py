"""
** test_me_2.py **
This is a script to test the Model_Drift visualization utility
"""

from levels.Model_Drift.compare_model_stats import *

# ==============================
def test_me():

	compare_model_stats("Baseline_MLP_init_v0", "Baseline_MLP_trained_v1")

if __name__ == "__main__":
	test_me()