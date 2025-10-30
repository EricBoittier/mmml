
# Ideas

# Use existing model


## Existing 
(base model) PhysnetMP2 


(transfer model) PhysnetMP2 -> CCSD(T) n unknown 

# New model
Bidiketones
Base model (n data = 111_000)



## Analysis

In basin:
- Normal Model Analysis
- DMC
- Optimization (BFGS)
- MD simulations (stability NVE initialized at different T)

Out of the basin:
- 1D bond scans, etc



- Distributions of features 
    - distibution shift (mean vs median)
    - during training
    - after convergence (eg. train all models to 1.0 kcal/mol validation)


## Tests  
        -- fold over shuffles of the dataset(s) (kFolds)
        -- by datasets (https://zenodo.org/records/4585449)
        -- [1] different molecule - same LoT
        -- [2] same molecule - different LoT

    -- test different weights in the loss function (E vs F)

            -- analogy - mean shift vs asymptotics

            Test hypothesis:
            energy pref -> less shift of the feature distribution during training
            forces pref -> larger shifts
            restarts, E-F -> larger shifts
                    F-E -> smaller dist. shifts.

https://zenodo.org/records/4585449/files/ch2o_cc_avtz_3601.npz?download=1

https://zenodo.org/records/4585449/files/ch2o_ccf12_avtz_3601.npz?download=1

https://zenodo.org/records/4585449/files/ch2o_mp2_avtz_3601.npz?download=1

https://zenodo.org/records/4585449/files/ch3cho_mp2_avtz_10073.npz?download=1

https://zenodo.org/records/4585449/files/ch3conh2_mp2_avtz_12601.npz?download=1

https://zenodo.org/records/4585449/files/ch3cooh_mp2_avtz_10910.npz?download=1

https://zenodo.org/records/4585449/files/ch3no2_mp2_avtz_9001.npz?download=1

https://zenodo.org/records/4585449/files/ch3oh_cc_avtz_7201.npz?download=1

https://zenodo.org/records/4585449/files/ch3oh_mp2_avtz_7201.npz?download=1

https://zenodo.org/records/4585449/files/hcooh_cc_avtz_5401.npz?download=1

https://zenodo.org/records/4585449/files/hcooh_ccf12_avtz_5401.npz?download=1

https://zenodo.org/records/4585449/files/hcooh_mp2_avtz_5401.npz?download=1

https://zenodo.org/records/4585449/files/hono_cc_avtz_6406.npz?download=1

https://zenodo.org/records/4585449/files/hono_ccf12_avtz_6406.npz?download=1

https://zenodo.org/records/4585449/files/hono_mp2_avtz_6406.npz?download=1




