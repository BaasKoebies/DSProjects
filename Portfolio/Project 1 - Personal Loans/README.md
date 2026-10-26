# Which customers to call about a personal loan

**To:** Naledi Mokoena, Head of Personal Lending, Karoo Bank
**Status:** Draft. Sections marked *To be written* depend on the model results.

Karoo Bank is fictional. The figures come from the public Thera Bank personal loan dataset (5,000 customers), relabelled for this memo.

## Decision

Which customers should Karoo Bank call about a personal loan offer, and what is that worth per 1,000 customers?

## Answer

*To be written after the model results.*

## Baseline

Each call costs R500, and each accepted loan earns R5,000 in margin. Both figures are assumptions. At those figures, a call pays for itself when the customer's chance of accepting is at least 10% (R500 / R5,000).

In the data, 9.6% of customers accepted the offer, so a model that predicts "no" for everyone is 90.4% accurate. This is why accuracy is not used to judge the models.

| Rule | Profit per 1,000 customers |
| --- | --- |
| Call nobody | R0 |
| Call everyone | -R20,000 |
| Call everyone above an income cutoff | *To be written* |
| Perfect foresight (call only the 96 who accept) | R432,000 |

Every customer in the data received the offer, so there is no control group. The data cannot show who would have borrowed without a call. The profit figures assume that a customer who is not called does not borrow.

## Method

*To be written.*

## Results

*To be written.*

## Sensitivity

*To be written.*

## Limitations

*To be written.*

## Reproduce

You need [uv](https://docs.astral.sh/uv/getting-started/installation/). uv installs Python 3.11 and the locked dependency versions for you.

1. Clone the repository and go to the project folder:

   ```bash
   git clone https://github.com/BaasKoebies/DSProjects.git
   cd "DSProjects/Portfolio/Project 1 - Personal Loans"
   ```

1. Create the environment from `uv.lock`:

   ```bash
   uv sync
   ```

1. Run the tests:

   ```bash
   uv run pytest
   ```

   All tests pass. If `test_csv_fingerprint_matches` fails on Windows, check that the `.gitattributes` file is present in the project folder, then check out the CSV again.

1. Run the exploratory analysis:

   ```bash
   uv run python -m loan_targeting.eda
   ```

   The script prints the data profile and saves the charts to `figures/`.
