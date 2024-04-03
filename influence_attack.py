import numpy as np

def influence_attack(
    X_train,
    y_train,
    X_test,
    y_test,
    x_lim_tuples,
    subpop_data,
    args,
    ScikitModel,
    num_poisons,
    num_steps,
    lr=1e-1,
):
    if args.model_type == "lr":
        raise ValueError("LR not yet supported for influence attack")
    if args.model_type != "svm":
        raise ValueError("Model type not recognized")

    trn_sub_x,trn_sub_y,trn_nsub_x,trn_nsub_y, \
        tst_sub_x,tst_sub_y,tst_nsub_x,tst_nsub_y = subpop_data
    
    # Choose poison seeds from target subpopulation
    np.random.seed(args.rand_seed)
    psn_ix = np.random.choice(
        trn_sub_x.shape[0],
        num_poisons,
        replace=True
    )
    X_psn = trn_sub_x[psn_ix]
    y_psn = -trn_sub_y[psn_ix]

    history = [(X_psn.copy(), y_psn.copy())]

    # Fit model
    fit_intercept = True
    X_cur = np.concatenate([X_train, X_psn], axis=0)
    y_cur = np.concatenate([y_train, y_psn], axis=0)
    C = 1.0 / (X_cur.shape[0] * args.weight_decay)
    model = ScikitModel(
        C=C,
        tol=1e-8,
        fit_intercept=fit_intercept,
        random_state=args.rand_seed,
        verbose=False,
        max_iter=32000
    ).fit(X_cur, y_cur)

    # Influence attack
    scores = []
    score = model.score(tst_sub_x, tst_sub_y)
    margin_tgt = -tst_sub_y * (np.dot(tst_sub_x, model.coef_.flatten()) + model.intercept_)
    loss_tgt = np.maximum(0, 1 - margin_tgt)
    scores.append((score, loss_tgt.mean()))

    for step in range(num_steps):
        # Fit model
        X_cur = np.concatenate([X_train, X_psn], axis=0)
        y_cur = np.concatenate([y_train, y_psn], axis=0)
        model.fit(X_cur, y_cur)

        theta = model.coef_.flatten()
        bias = model.intercept_

        # Compute margins
        margin_psn = y_psn * (np.dot(X_psn, theta) + bias)
        margin_tgt = -tst_sub_y * (np.dot(tst_sub_x, theta) + bias)
        loss_psn = np.maximum(0, 1 - margin_psn)
        loss_tgt = np.maximum(0, 1 - margin_tgt)
        sv_psn = loss_psn > 1.0
        sv_tgt = loss_tgt > 1.0
        if np.sum(sv_tgt) == 0:
            break

        # Descent step
        dx_psn = np.where(sv_psn, -y_psn, 0)
        dw_sub = tst_sub_y[:, None] * tst_sub_x
        dw_sub = dw_sub[sv_tgt].sum(axis=0, keepdims=True) / tst_sub_x.shape[0]
        # can omit the bias term, since does not affect dx_psn

        X_grad = lr * dx_psn[:, None] * dw_sub
        if np.linalg.norm(X_grad) < 1e-8:
            break
            
        X_psn += X_grad
        pos_psn = y_psn == 1
        neg_psn = y_psn == -1
        X_psn[pos_psn] = np.clip(X_psn[pos_psn], x_lim_tuples[0][0], x_lim_tuples[0][1])
        X_psn[neg_psn] = np.clip(X_psn[neg_psn], x_lim_tuples[1][0], x_lim_tuples[1][1])

        # Evaluate model on target subpopulation
        score = model.score(tst_sub_x, tst_sub_y)
        scores.append((score, loss_tgt.mean()))
        history.append((X_psn.copy(), y_psn.copy()))
        print("Step {0}: score={1}, loss={2}".format(step, score, loss_tgt.mean()))

    best_ix = scores.index(min(scores)) # lexicographic on (err, adv loss)
    X_psn, y_psn = history[best_ix]

    return X_psn, y_psn, history