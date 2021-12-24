Mushroom classification continuous training, served with Streamlit


Level 0:

```
┌────────────┐     ┌────┐     ┌────────┐      ┌────────┐
│Data grab   ├────►│Data├────►│ML Model├─────►│Model   │
│from Twitter│     │Prep│     │Training│      │Artifact│
└────────────┘     └────┘     └────────┘      └────┬───┘
                                                   │
                                                   │
                                                   │
                                              ┌────▼────┐
                                              │Streamlit│
                                              │serving  │
                                              └─────────┘
```