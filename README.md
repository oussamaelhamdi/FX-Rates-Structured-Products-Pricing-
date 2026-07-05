# FX Structured Products Pricer

React + Vite single page app pricing FX structured products on EURUSD under Black Scholes:
TARF family (Vanilla, Liability Knock Out, EKI, Pivot, EKI Pivot), Accumulator
(European / American KO, rolling / ZC settlement), Dual Currency Deposit and Vanilla Options.
Monte Carlo engine with antithetic variates and common random numbers for Greeks;
closed form Garman Kohlhagen for DCD and vanillas.

## Run locally
    npm install
    npm run dev

## Build
    npm run build        # output in dist/

## Deploy on Vercel
Option A (Git): push this folder to a GitHub/GitLab repo, then "Add New Project" on
vercel.com and import it. Vercel auto detects Vite; defaults are already pinned in
vercel.json (build: npm run build, output: dist). Just click Deploy.

Option B (CLI):
    npm i -g vercel
    vercel           # first deploy (preview)
    vercel --prod    # production
