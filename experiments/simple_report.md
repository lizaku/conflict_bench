# Simple report

- model: `meta-llama/Llama-3.1-8B-Instruct`
- items: 400 over 40 relations
- context-following rate: 0.470

```
         method      axis  auroc  auroc_base  accuracy  acc_majority  flip_rate  flip_flippable  specific_effect  factor         target   n                                           note
         p_true detection  0.590         0.5       NaN           NaN        NaN             NaN              NaN     NaN            NaN 400                                               
   linear_probe detection  0.570         0.5       NaN           NaN        NaN             NaN              NaN     NaN            NaN 400                                               
    context_jsd detection  0.530         0.5       NaN           NaN        NaN             NaN              NaN     NaN            NaN 400                                               
  diffmean_proj detection  0.522         0.5       NaN           NaN        NaN             NaN              NaN     NaN            NaN 400                                               
     context_kl detection  0.521         0.5       NaN           NaN        NaN             NaN              NaN     NaN            NaN 400                                               
     logit_lens detection  0.486         0.5       NaN           NaN        NaN             NaN              NaN     NaN            NaN 400 control: readout at the probes' layer/position
         margin detection  0.472         0.5       NaN           NaN        NaN             NaN              NaN     NaN            NaN 400                           floor: the DV itself
confidence_gain detection  0.466         0.5       NaN           NaN        NaN             NaN              NaN     NaN            NaN 400                                               
            bow detection  0.453         0.5       NaN           NaN        NaN             NaN              NaN     NaN            NaN 400                        control: text-only null
            cad  steering    NaN         NaN       NaN           NaN      0.975           0.990           44.677     2.0 use_parametric 400                                               
         adacad  steering    NaN         NaN       NaN           NaN      0.935           0.949           30.401     2.0 use_parametric 400                                               
        act_add  steering    NaN         NaN       NaN           NaN      0.010           0.667            2.262     1.0    use_context 400                                               
            caa  steering    NaN         NaN       NaN           NaN      0.010           0.667           -6.521     0.5    use_context 400                                               
prompt_instruct  steering    NaN         NaN       NaN           NaN      0.002           0.167           -2.150     0.0    use_context 400                            baseline: prompting
         ckplug  steering    NaN         NaN       NaN           NaN      0.028           0.028            1.193     1.0 use_parametric 400                                               
```

## Reading it

- **detection** - `auroc` is over relations held out by GroupKFold; `accuracy` uses a threshold fitted on the train folds only.
- a detector is only doing something if it beats all three of `auroc_base` (its own row), the `bow` row, and the `logit_lens` row.
- **steering** - `flip_rate` is the share of items whose margin changed sign; `flip_flippable` conditions on the items that could move, and is the number to compare across methods.
- `factor` is where each method peaked, so the doses are not comparable between rows - `flip_flippable` is.
- `specific_effect` is delta-margin minus the matched control; a flip rate with a near-zero specific effect is the control moving, not the method.
