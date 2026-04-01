# Automatic Prefix Caching in Omni Models


---

## Table of Contents

- [Overview](#overview)
- [High-Level Approach](#high-level-approach)
- [Example](#example)
- [Current Limitations](#current-limitations)

---

### Overview

Prefix caching in the context of kv-cache management is a useful optimization for avoiding redundant computations. The main idea is that we store portions of the kv-cache from processed requests, so that we can reuse them if incoming requests have the same prefix as previous requests.

vLLM manages the kv-cache as blocks, which represent a span of tokens of a fixed length. Blocks are hashable by the content that they contain, which typically means the tokens within the span, but also could be influenced by other factors, e.g., LoRA and multimodal data.

vLLM implements automatic prefix caching for managing its kv-cache, which is best understood by reading the design document [here](https://docs.vllm.ai/en/latest/design/prefix_caching/). vLLM-Omni builds on top of the prefix caching mechanism in a noninvasive way to allow caching between stages in Omni pipelines. This typically means for a given stage we aim to support caching for the following:

- The last hidden states produced by the stage
- Model / stage specific multimodal data

### High-Level Approach
!!! note "Note 1"
    Prior to reading this section, it's recommended to take a look at the design documents in vLLM for [Automatic Prefix Caching](https://docs.vllm.ai/en/latest/features/automatic_prefix_caching/), which will make some of the concepts more clear.

The main focus of vLLM-Omni's approach to prefix caching stage outputs is to build on vLLM's prefix caching in the least invasive way possible while minimizing impact for cache misses, and consuming a minimal amount of GPU memory. To understand the implementation, there are a few important things to note:

- Between stages, device tensors are generally moved to CPU; this is important since we're just caching the outputs of stages, so it is okay to keep the entire cache on the CPU.

- For a tensor to be considered cacheable, the first dimension (currently) needs to be the same as the token count, as it allows us to reuse block/slot mappings for our externally maintained tensor caches.

With this in mind, consider the set of blocks in a 2D layout, where the row represents the index of blocks being considered, and the columns represent the slots corresponding to tokens within each block. Since we know the `num_blocks` and `block_size` from our kv cache config, if we want to cache a tensor with feature size `D`, we can preallocate a CPU tensor of size `(num_blocks, block_size, D)`, and use the same block index and slot mapping to retrieve the corresponding feature vector.


### Example
!!! note "Note 2"
    Prefix caching in vLLM Omni currently is only supported on AutoRegressive stages with one kv-cache group.

The way in which vLLM Omni ties into vLLM's prefix caching is best understood by example. Say that we have the following:

- `num_blocks=8`
- `block_size=4`
- `hidden_size=2`
- A stage specific multimodal output `mm_feature` with feature dimension `16`

The prefix cache flow is then outlined below.

1. When the model is initialized, we can determine the `hidden_size` from the `ModelConfig`, and allocate a cache of size `(num_blocks, block_size, hidden_size)`.

2. Say we process the request `The quick brown fox was tired and slept beneath the shady tree`, which is 12 tokens and evenly divides into 3 blocks as shown below.

```
         [  The quick brown fox  ] [  was tired and slept ] [beneath the shady tree ]
Block 1: |<--- block tokens ---->|
Block 2: |<------- prefix ------>| |<--- block tokens --->|
Block 3: |<------------------ prefix -------------------->| |<--- block tokens ---->|
```

When the request processes, we inspect the multimodal outputs and identify the `mm_feature` tensor, which will be of shape `(seq_len, feature_dim)`, i.e., `(12, 16)` in this example. We note that the first axis is dependent on the `seq_len` and add a new cache_tensor of shape `(num_blocks, block_size, feature_dim)` to our multimodal cache for tensors.


3. If we lay out the cache as a 2D tensor of shape (`num_blocks`, `block_size`), we'll have something like the following:

```
0: [  The quick brown fox  ]
1: [  was tired and slept  ]
2: [beneath the shady tree ]
3: [EMPTY]
...
7: [EMPTY]
```

Or, if we flatten it down to 1D,
```
0: The
1: quick
2: brown
3: fox
...
11: tree
12: [EMPTY]
...
```

which we can think of as row indices into the hidden states tensor if we view it as the 2D shape `(num_blocks x block_size, feature_dim)`. That is, the analogous flattened (from 3D -> 2D) mapping of the cache for hidden states becomes the following.
```
0: <hidden states vector of len 2 corresponding to 'The'>
1: <hidden states vector of len 2 corresponding to 'quick'>
2: <hidden states vector of len 2 corresponding to 'brown'>
3: <hidden states vector of len 2 corresponding to 'fox'>
...
11: <hidden states vector of len 2 corresponding to 'tree'>
12: [EMPTY]
...
```

Similarly, for the multimodal outputs cache, the flattened coordinates are the same, but the `mm_feature` maps to vectors of length `16` instead of the hidden size of `2`.


4. Now, say that we receive a new request `The quick brown fox jumped over the dog`.

```
         [  The quick brown fox  ] [  jumped over the dog ]
Block 1: |<--- block tokens ---->|
Block 2: |<------- prefix ------>| |<--- block tokens --->|
```

Here, we will have a cache hit for `Block 1` which will be detected by vLLM based on the hash of the first block when it's handling the prefix caching on the kv-cache. As a result, when we get the output from the scheduler, we will see that `num_computed_tokens=4` (corresponding to the cached first block), and we only need to process the remaining 4 new tokens in the new prefill.

Since we have the block indices / slot mappings from the kv cache manager, we can simply mirror the mappings and leverage the same indices for the cached hidden states and multimodal outputs. This allows us to look up the correct tensors from our externally maintained 3D caches.

```
0: [  The quick brown fox  ] < already in the cache
1: [  was tired and slept  ]
2: [beneath the shady tree ]
3: [ jumped over the dog  ] < added on the second request
4: [EMPTY]
...
7: [EMPTY]
...
```

Finally, to pass the full hidden states and multimodal outputs to the next stage, we simply concatenate the cached contents with the corresponding new tensors computed from the current forward call.
