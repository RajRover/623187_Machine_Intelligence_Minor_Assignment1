# California Housing: What the Data Can Tell Us

*One-page summary prepared for Problem 5(c) — addressed to a reader with no machine learning background*

---

## What Is This Dataset?

The California Housing dataset records information about **20,640 neighbourhood blocks** across California, collected in the 1990 US Census. For each block it tells us things like the typical income of residents, the age of the houses, how many rooms they have, and how many people live in them. The goal is to understand what makes house prices go up or down across the state.

---

## The Two Most Important Data-Quality Problems

### 1. The price ceiling — values above $500,000 are hidden

When you look at the distribution of house prices, you notice something odd: a large pile of blocks are all recorded at exactly $500,000. This is not because hundreds of neighbourhoods happen to have exactly the same median price. It is because the original survey **stopped counting above $500,000** — any block that was actually worth more was simply recorded as $500k. Statisticians call this *right-censoring* (or *top-coding*).

Why does it matter? If we train a model on this data, it will learn that no neighbourhood is worth more than $500,000 — even wealthy areas like Pacific Heights or Beverly Hills. The model will therefore **systematically underestimate prices in expensive districts**, sometimes by hundreds of thousands of dollars.

### 2. Suspicious extreme values in room and occupancy counts

Three columns — *average rooms per household*, *average bedrooms per household*, and *average occupants per household* — contain values that defy common sense. For example, some blocks appear to average over 100 rooms per household, or 100 occupants. These are almost certainly errors caused by blocks with just one or two households: dividing the total rooms by a very small number produces an absurdly large average.

If we leave these extreme values in, the model would treat a block with 120 average rooms the same as any other input, distorting its picture of what a "typical" neighbourhood looks like. The right fix is to cap these values at a reasonable upper limit (for example, the 99th percentile of the data).

---

## Three Predictors I Would Use First

**1. Median Income** — This is by far the strongest single indicator of house price. Wealthier residents live in more expensive areas; the relationship is consistent across the whole state.

**2. Latitude and Longitude (combined)** — Geography matters enormously. The San Francisco Bay Area and the Greater Los Angeles basin are clearly visible on a map as two dense pockets of very high house values. No single number captures this, but together the coordinates tell the model *where* a block sits, which carries a huge amount of information about local demand and desirability.

**3. Housing Median Age** — Older housing stock in prime urban locations (San Francisco, Santa Barbara) tends to be expensive, while new developments in inland areas tend to be cheaper. Age is not a perfect predictor, but it adds context that income alone misses.

---

*This summary was prepared as part of EC-417 Assignment 1, NIT Andhra Pradesh, August 2026.*
