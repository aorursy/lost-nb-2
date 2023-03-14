#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_cell_magic('script', 'false', '\n# libraries used\n# --------------\nusing XGBoost\nusing DataFrames\nusing CSV \nusing StatsBase\nusing DelimitedFiles\n\n# initial setup\n@info "Set Working Directory..."\ncd(joinpath(homedir(), "Downloads/santander-customer-transaction-prediction"))\n\nfunction read_data()\n    @info "Input Data..."\n    train_df = CSV.read("train.csv")\n    test_df = CSV.read("test.csv")\n    return(train_df, test_df)\nend\n\ntrain_df, test_df = read_data()\n\nidx = [c for c in names(train_df) if c != :ID_code && c != :target]\n\n# XGBoost\n# -------\nnum_rounds = 500\n\n# define train sets \n# -----------------\ntrain_x = convert(Array{Float32}, train_df[idx])\ntrain_y = convert(Array{Int32}, train_df[:target])\n\n# define test set\n# ---------------\ntest_x = convert(Array{Float32}, test_df[idx])\n\ndtrain = DMatrix(train_x, label = train_y)\n\nboost = xgboost(dtrain, num_rounds, eta = .03, objective = "binary:logistic")\n\nprediction = XGBoost.predict(boost, test_x)\n\nprediction_rounded = Array{Int64, 1}(map(val -> round(val), prediction))\n\nsub = hcat(test_df[:ID_code], prediction)\nsub = DataFrame(sub)\n\n# clean up submission\n# -------------------\nrename!(sub, :x1 => :ID_code)\nrename!(sub, :x2 => :target)\n\nCSV.write("predictions.csv", sub)')

