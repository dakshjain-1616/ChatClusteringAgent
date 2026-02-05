# NEO User Chat Analysis Report

**Generated:** 2026-02-05T08:03:17.056501

---

## Executive Summary

- **Total Conversations Analyzed:** 50
- **Total User Messages:** 128
- **User Satisfaction Index:** 24.53%
- **Topics Identified:** 5

## User Satisfaction Analysis

### Overall User Satisfaction Index: 24.53%

#### Satisfaction Distribution

- **High Satisfaction** (≥70%): 6 conversations (12.0%)
- **Medium Satisfaction** (40-70%): 7 conversations (14.0%)
- **Low Satisfaction** (<40%): 37 conversations (74.0%)

#### Sentiment Distribution

- **Positive Messages:** 36 (28.1%)
- **Negative Messages:** 92 (71.9%)

#### Top 5 Most Satisfied Conversations

1. Chat ID: `7bb5fad6-df7a-46...` - Score: 99.82% (1 positive, 0 negative)
2. Chat ID: `c7875099-76e2-40...` - Score: 99.65% (1 positive, 0 negative)
3. Chat ID: `13730c6e-94be-4f...` - Score: 98.74% (1 positive, 0 negative)
4. Chat ID: `3f7a0525-ce67-49...` - Score: 98.74% (2 positive, 0 negative)
5. Chat ID: `96b83a20-dfc9-49...` - Score: 95.08% (3 positive, 0 negative)

#### Top 5 Least Satisfied Conversations

1. Chat ID: `a2016c77-33c9-4b...` - Score: 0.07% (0 positive, 1 negative)
2. Chat ID: `1fe55c4f-f604-4e...` - Score: 0.16% (0 positive, 1 negative)
3. Chat ID: `b5308d4a-3f32-4e...` - Score: 0.39% (0 positive, 1 negative)
4. Chat ID: `03a29b14-2d6a-4d...` - Score: 0.39% (0 positive, 1 negative)
5. Chat ID: `34e8f762-7944-45...` - Score: 0.45% (0 positive, 3 negative)

---

## Topic Clusters

### Topic 4

- **Conversations:** 14 (28.0%)
- **Keywords:** se, eine, sein, website, nicht
- **Sample Chats:** `e904d6f7-f64...`, `d813e415-654...`, `0306c6f9-1ff...`

### Topic 3

- **Conversations:** 12 (24.0%)
- **Keywords:** ai, make, snoring, دون, section
- **Sample Chats:** `ea704b7e-ec5...`, `a2016c77-33c...`, `3f7a0525-ce6...`

### Topic 1

- **Conversations:** 12 (24.0%)
- **Keywords:** app, create, using, use, project
- **Sample Chats:** `c3cf0563-48e...`, `44312dfa-b55...`, `4bdfdcc5-0c4...`

### Topic 0

- **Conversations:** 7 (14.0%)
- **Keywords:** python, create, data, vjerovatnoćom kumulativni, uticaj kroz
- **Sample Chats:** `1fe55c4f-f60...`, `b5308d4a-3f3...`, `03a29b14-2d6...`

### Topic 2

- **Conversations:** 5 (10.0%)
- **Keywords:** make, want, request, upload, css
- **Sample Chats:** `431f0827-733...`, `00b1e9c3-918...`, `5ebc21f6-dc2...`

## Message Type Distribution

**Total Messages Analyzed:** 128

- **General:** 45 messages (35.2%)
- **Commands:** 38 messages (29.7%)
- **Questions:** 36 messages (28.1%)
- **Requests:** 8 messages (6.2%)
- **Feedback:** 1 messages (0.8%)

---

## Methodology

- **Clustering Method:** KMeans with SentenceTransformer embeddings
- **Embedding Model:** all-MiniLM-L6-v2
- **Sentiment Model:** distilbert-base-uncased-finetuned-sst-2-english
- **Hardware:** GPU-accelerated (if available)
