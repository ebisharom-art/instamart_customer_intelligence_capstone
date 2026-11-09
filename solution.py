
import os, re, json, time, math, random, sqlite3, argparse
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, confusion_matrix, precision_score, recall_score, f1_score, accuracy_score
from sklearn.decomposition import PCA
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import joblib
import mock_llm

ROOT=Path(__e if False else ".")
for d in ["data/raw","data/clean","data/policies","outputs","sql","state"]:
    Path(d).mkdir(parents=True,exist_ok=True)

def part_a():
    rawc=pd.read_csv("data/raw/customers.csv",keep_default_na=False)
    rawp=pd.read_csv("data/raw/products.csv")
    rawo=pd.read_csv("data/raw/orders.csv",keep_default_na=False)
    # Audit summaries
    def table_audit(frame):
        miss=frame.isna() | frame.astype(str).apply(lambda col: col.str.strip().eq(""))
        return {"shape":list(frame.shape),"dtypes":{k:str(v) for k,v in frame.dtypes.items()},
                "missing":{k:{"count":int(miss[k].sum()),"pct":round(float(miss[k].mean()*100),2)} for k in frame.columns}}
    audit={"customers_table":table_audit(rawc),"products_table":table_audit(rawp),"orders_table":table_audit(rawo),
      "customers_shape":list(rawc.shape),"products_shape":list(rawp.shape),"orders_shape":list(rawo.shape),
      "orders_exact_duplicates":int(rawo.duplicated().sum()),"orders_pending":int((rawo.status=="pending").sum()),
      "distinct_order_id":int(rawo.order_id.nunique()),"non_numeric_amount":int(pd.to_numeric(rawo.order_amount,errors="coerce").isna().sum()),
      "negative_delivery":int((pd.to_numeric(rawo.delivery_minutes,errors="coerce")<0).sum()),
      "blank_delivery":int((rawo.delivery_minutes.astype(str).str.strip()=="").sum()),
      "blank_age":int((rawc.age.astype(str).str.strip()=="").sum()),
      "age_outside_10_100":int(((pd.to_numeric(rawc.age,errors="coerce")<10)|(pd.to_numeric(rawc.age,errors="coerce")>100)).sum()),
      "raw_city_spellings":int(rawc.city.nunique()),"raw_gender_spellings":int(rawc.loc[rawc.gender.astype(str).str.strip()!="","gender"].nunique()),
      "blank_gender":int((rawc.gender.astype(str).str.strip()=="").sum())}
    # customers
    c=rawc.copy()
    c["city"]=c.city.astype(str).str.strip().str.title()
    c["gender"]=c.gender.astype(str).str.strip().str.lower().replace({"m":"male","f":"female","":"unknown"})
    c["gender"]=c.gender.where(c.gender.isin(["male","female"]), "unknown")
    c["age"]=pd.to_numeric(c.age,errors="coerce")
    c.loc[~c.age.between(10,100),"age"]=np.nan
    c["age"]=c["age"].fillna(c.groupby("city")["age"].transform("median"))
    c["signup_date"]=pd.to_datetime(c.signup_date)
    # products
    p=rawp.copy()
    # orders
    o=rawo.copy()
    o["order_ts"]=pd.to_datetime(o.order_ts)
    o=o.drop_duplicates()
    o=o.sort_values("order_ts").drop_duplicates("order_id",keep="last").copy()
    o["quantity"]=pd.to_numeric(o.quantity,errors="coerce")
    o["discount_pct"]=pd.to_numeric(o.discount_pct,errors="coerce")
    o["order_amount"]=pd.to_numeric(o.order_amount,errors="coerce")
    o=o.merge(p[["product_id","price_inr","category"]],on="product_id",how="left")
    miss=o.order_amount.isna()
    o.loc[miss,"order_amount"]=(o.loc[miss,"price_inr"]*o.loc[miss,"quantity"]*(1-o.loc[miss,"discount_pct"]/100)).round(2)
    o["delivery_minutes"]=pd.to_numeric(o.delivery_minutes,errors="coerce")
    o.loc[o.delivery_minutes<0,"delivery_minutes"]=np.nan
    comp=o.status.eq("completed")
    med=o.loc[comp].groupby("category").delivery_minutes.transform("median")
    o.loc[comp,"delivery_minutes"]=o.loc[comp,"delivery_minutes"].fillna(med)
    o=o.drop(columns=["price_inr","category"])
    c.to_csv("data/clean/customers.csv",index=False)
    p.to_csv("data/clean/products.csv",index=False)
    o.to_csv("data/clean/orders.csv",index=False)
    with sqlite3.connect("data/instamart.db") as con:
        c.to_sql("customers",con,if_exists="replace",index=False)
        p.to_sql("products",con,if_exists="replace",index=False)
        oo=o.copy(); oo["order_ts"]=oo.order_ts.dt.strftime("%Y-%m-%d %H:%M:%S")
        oo.to_sql("orders",con,if_exists="replace",index=False)
    completed=o[o.status=="completed"]
    audit.update({"clean_orders":len(o),"completed_orders":len(completed),"cancelled_orders":int((o.status=="cancelled").sum()),
      "completed_revenue":round(float(completed.order_amount.sum()),2),
      "customer_city_counts":c.city.value_counts().to_dict(),"customer_gender_counts":c.gender.value_counts().to_dict()})
    # SQL queries
    sqls=[
"""-- Q1
SELECT customer_id, SUM(order_amount) AS spend FROM orders WHERE status='completed' AND order_amount>500 GROUP BY customer_id HAVING SUM(order_amount)>8000;""",
"""-- Q2
SELECT c.city,p.category,SUM(o.order_amount) revenue FROM orders o JOIN customers c USING(customer_id) JOIN products p USING(product_id) WHERE o.status='completed' GROUP BY c.city,p.category;""",
"""-- Q3
SELECT product_id,product_name,price_inr FROM products WHERE price_inr>(SELECT AVG(price_inr) FROM products);""",
"""-- Q4
SELECT c.customer_id FROM customers c LEFT JOIN orders o ON c.customer_id=o.customer_id AND o.status='completed' WHERE o.order_id IS NULL;""",
"""-- Q5
WITH spend AS (SELECT customer_id,SUM(order_amount) total_spend FROM orders WHERE status='completed' GROUP BY customer_id), avg_spend AS (SELECT AVG(total_spend) avg_total FROM spend) SELECT * FROM spend WHERE total_spend>(SELECT avg_total FROM avg_spend);""",
"""-- Q6
WITH spend AS (SELECT c.customer_id,c.city,SUM(o.order_amount) total_spend FROM orders o JOIN customers c USING(customer_id) WHERE o.status='completed' GROUP BY c.customer_id,c.city) SELECT *,RANK() OVER(PARTITION BY city ORDER BY total_spend DESC) rnk,DENSE_RANK() OVER(PARTITION BY city ORDER BY total_spend DESC) dense_rnk,ROW_NUMBER() OVER(PARTITION BY city ORDER BY total_spend DESC) row_num FROM spend;""",
"""-- Q7
WITH daily AS (SELECT substr(order_ts,1,10) day,SUM(order_amount) revenue FROM orders WHERE status='completed' GROUP BY substr(order_ts,1,10)) SELECT day,revenue,SUM(revenue) OVER(ORDER BY day) cumulative_revenue FROM daily;"""
]
    Path("sql/queries.sql").write_text("\n\n".join(sqls),encoding="utf-8")
    qresults=[]
    with sqlite3.connect("data/instamart.db") as con:
        for q in sqls:
            qresults.append(len(pd.read_sql_query(q.split(";")[0].split("\n",1)[-1],con)))
    # exploratory stats
    x=completed.order_amount
    q1,q3=x.quantile([.25,.75]); iqr=q3-q1
    stats={"mean":float(x.mean()),"median":float(x.median()),"skew":float(x.skew()),"q1":float(q1),"q3":float(q3),
           "lower_fence":float(q1-1.5*iqr),"upper_fence":float(q3+1.5*iqr),
           "outliers_below":int((x<q1-1.5*iqr).sum()),"outliers_above":int((x>q3+1.5*iqr).sum())}
    audit["order_stats"]=stats
    audit["sql_result_rows"]=qresults
    joined=completed.merge(c[["customer_id","city"]],on="customer_id").merge(p[["product_id","category"]],on="product_id")
    pivot=pd.pivot_table(joined,index="city",columns="category",values="order_amount",aggfunc="sum",margins=True,fill_value=0)
    audit["pivot_grand_total"]=float(pivot.loc["All","All"])
    audit["city_spend"]=completed.merge(c[["customer_id","city"]],on="customer_id").groupby("city").order_amount.agg(["sum","mean","count"]).round(2).to_dict("index")
    Path("outputs/part_a_audit.json").write_text(json.dumps(audit,indent=2,default=str))
    print("PART A",json.dumps(audit,default=str))
    return c,o,p,audit

def part_b(c,o,p):
    comp=o[o.status=="completed"].copy()
    snap=pd.Timestamp("2025-10-01")
    rfm=comp.groupby("customer_id").agg(last=("order_ts","max"),frequency=("order_id","nunique"),monetary=("order_amount","sum")).reset_index()
    rfm["recency"]=(snap-rfm["last"]).dt.days
    feats=["recency","frequency","monetary"]
    scaler=StandardScaler(); Z=scaler.fit_transform(rfm[feats])
    ks=[]
    for k in range(2,9):
        km=KMeans(n_clusters=k,n_init=10,random_state=42).fit(Z)
        ks.append({"k":k,"inertia":km.inertia_,"silhouette":silhouette_score(Z,km.labels_)})
    # Select K=4 as strongest silhouette/elbow in project reference
    km=KMeans(n_clusters=4,n_init=10,random_state=42).fit(Z)
    rfm["segment_id"]=km.labels_
    profiles=rfm.groupby("segment_id").agg(size=("customer_id","size"),recency=("recency","mean"),frequency=("frequency","mean"),monetary=("monetary","mean")).reset_index()
    names={}
    for _,row in profiles.iterrows():
        if row.monetary==profiles.monetary.max(): name="Premium"
        elif row.frequency==profiles.frequency.max(): name="Frequent"
        elif row.recency==profiles.recency.min(): name="Recent"
        else: name="Occasional"
        names[int(row.segment_id)]=name
    rfm["segment_name"]=rfm.segment_id.map(names)
    rfm["is_premium"]=rfm.segment_name.eq("Premium")
    rfm.to_csv("outputs/customer_segments.csv",index=False)
    pca=PCA(n_components=2).fit(Z)
    # Lapse cohort and leakage-safe features
    cutoff=pd.Timestamp("2025-09-01")
    before=o[(o.order_ts<cutoff)&(o.status.isin(["completed","cancelled"]))].copy()
    compb=before[before.status=="completed"]
    cohort_ids=compb.customer_id.unique()
    y_lapse=comp.groupby("customer_id").apply(lambda g:int(not ((g.order_ts>=cutoff)&(g.order_ts<pd.Timestamp("2025-10-01"))).any()),include_groups=False)
    feat_rows=[]
    for cid in cohort_ids:
        g=before[before.customer_id==cid]; gc=g[g.status=="completed"]
        if gc.empty: continue
        custrow=c[c.customer_id==cid].iloc[0]
        cat=gc.merge(p[["product_id","category"]],on="product_id").category.mode().iloc[0]
        feat_rows.append({"customer_id":cid,"recency":(cutoff-gc.order_ts.max()).days,"frequency":gc.order_id.nunique(),
          "monetary":gc.order_amount.sum(),"avg_discount":gc.discount_pct.mean(),
          "cancel_rate":(g.status=="cancelled").sum()/len(g),"avg_delivery":gc.delivery_minutes.mean(),
          "favourite_category":cat,"city":custrow.city,"age":custrow.age,"gender":custrow.gender,
          "tenure":(cutoff-custrow.signup_date).days,"lapsed":int(not ((comp.customer_id==cid)&(comp.order_ts>=cutoff)&(comp.order_ts<pd.Timestamp("2025-10-01"))).any())})
    df=pd.DataFrame(feat_rows).dropna()
    y=df.pop("lapsed").astype(int); ids=df.pop("customer_id")
    num=["recency","frequency","monetary","avg_discount","cancel_rate","avg_delivery","age","tenure"]
    catcols=["favourite_category","city","gender"]
    X=df[num+catcols]
    assert not X.isna().any().any(), "Feature matrix contains missing values"
    Xtr,Xte,ytr,yte, idtr,idte=train_test_split(X,y,ids,test_size=.2,stratify=y,random_state=42)
    pre=ColumnTransformer([("num",Pipeline([("imp",SimpleImputer(strategy="median")),("scale",StandardScaler())]),num),
       ("cat",Pipeline([("imp",SimpleImputer(strategy="most_frequent")),("ohe",OneHotEncoder(drop="first",handle_unknown="ignore"))]),catcols)])
    logpipe=Pipeline([("pre",pre),("model",LogisticRegression(max_iter=1000))])
    loggrid=GridSearchCV(logpipe,{"model__C":[.01,.1,1,10,100]},cv=5,scoring="f1",n_jobs=-1).fit(Xtr,ytr)
    rfpipe=Pipeline([("pre",pre),("model",RandomForestClassifier(n_estimators=100,random_state=42))])
    rfgrid=GridSearchCV(rfpipe,{"model__max_depth":[3,5,8],"model__min_samples_leaf":[5,10,20]},cv=5,scoring="f1",n_jobs=-1).fit(Xtr,ytr)
    best=rfgrid.best_estimator_
    # Majority-class baseline on untouched test rows.
    baseline=np.zeros(len(yte),dtype=int)
    baseline_metrics={"accuracy":accuracy_score(yte,baseline),"lapsed_f1":f1_score(yte,baseline,zero_division=0)}
    pred=best.predict(Xte)
    cm=confusion_matrix(yte,pred).tolist()
    metrics={"precision":precision_score(yte,pred,zero_division=0),"recall":recall_score(yte,pred,zero_division=0),"f1":f1_score(yte,pred,zero_division=0)}
    joblib.dump(best,"outputs/lapse_model.joblib")
    loaded=joblib.load("outputs/lapse_model.joblib")
    assert np.array_equal(pred,loaded.predict(Xte))
    probs=best.predict_proba(Xte)[:,1]
    # Training-only random oversampling for logistic regression.
    train_df=Xtr.copy(); train_y=ytr.copy()
    counts=train_y.value_counts(); minority=int(counts.idxmin()); target=int(counts.max())
    minority_rows=train_df[train_y==minority]
    add=minority_rows.sample(n=target-int(counts.min()),replace=True,random_state=42)
    balanced_X=pd.concat([train_df,add],axis=0)
    balanced_y=pd.concat([train_y,pd.Series([minority]*len(add),index=add.index)])
    balanced_log=joblib.loads if False else loggrid.best_estimator_
    balanced_log.fit(balanced_X,balanced_y)
    bpred=balanced_log.predict(Xte)
    balanced_metrics={"precision":precision_score(yte,bpred,zero_division=0),"recall":recall_score(yte,bpred,zero_division=0),"f1":f1_score(yte,bpred,zero_division=0)}
    # Feature importances and interpretable logistic coefficient signs.
    rf_model=best.named_steps["model"]
    feature_names=best.named_steps["pre"].get_feature_names_out()
    top_idx=np.argsort(rf_model.feature_importances_)[::-1][:5]
    top_features=[{"feature":str(feature_names[i]),"importance":float(rf_model.feature_importances_[i])} for i in top_idx]
    log_model=loggrid.best_estimator_.named_steps["model"]
    log_names=loggrid.best_estimator_.named_steps["pre"].get_feature_names_out()
    coef_order=np.argsort(np.abs(log_model.coef_[0]))[::-1][:3]
    log_coefs=[{"feature":str(log_names[i]),"coefficient":float(log_model.coef_[0][i])} for i in coef_order]
    prem=rfm[rfm.is_premium].set_index("customer_id")
    ret=pd.DataFrame({"customer_id":idte.values,"lapse_probability":probs,"actually_lapsed":yte.values})
    ret=ret[ret.customer_id.isin(prem.index)].sort_values("lapse_probability",ascending=False).head(10)
    ret.to_csv("outputs/retention_list.csv",index=False)
    report={"rfm_customers":len(rfm),"recency_median":float(rfm.recency.median()),"frequency_mean":float(rfm.frequency.mean()),
      "monetary_mean":float(rfm.monetary.mean()),"kmeans":ks,"segment_profiles":profiles.to_dict("records"),
      "pca_variance":pca.explained_variance_ratio_.tolist(),"cohort":len(df),"lapsed":int(y.sum()),
      "train":len(Xtr),"test":len(Xte),"logistic_best":str(loggrid.best_params_),"logistic_cv_f1":loggrid.best_score_,
      "rf_best":str(rfgrid.best_params_),"rf_cv_f1":rfgrid.best_score_,"rf_confusion":cm,"rf_metrics":metrics,
      "baseline":baseline_metrics,"oversampled_logistic":balanced_metrics,"rf_top_features":top_features,"logistic_coefficients":log_coefs,
      "train_scaler_frequency_mean":float(best.named_steps["pre"].named_transformers_["num"].named_steps["scale"].mean_[1]),
      "full_cohort_frequency_mean":float(X.frequency.mean()),
      "retention_list_true_lapsed":int(ret.actually_lapsed.sum())}
    Path("outputs/part_b_report.json").write_text(json.dumps(report,indent=2,default=str))
    print("PART B",json.dumps(report,default=str))
    return rfm,report

def part_c():
    tickets=pd.read_csv("data/raw/tickets.csv",keep_default_na=False)
    seg=pd.read_csv("outputs/customer_segments.csv")
    patterns=[(re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+"),"[EMAIL]"),(re.compile(r"\b[6-9]\d{9}\b"),"[PHONE]"),(re.compile(r"ORD-\d{6}"),"[ORDER_ID]")]
    zero="CLASSIFY. Allowed categories: billing, delivery, technical, fraud. Reply JSON only: {\"category\": \"...\", \"reason\": \"...\"}."
    few=zero+"\nEXAMPLES:\nFraud example: unauthorized card transaction -> {\"category\":\"fraud\",\"reason\":\"unauthorized\"}\nDelivery example: parcel late -> {\"category\":\"delivery\",\"reason\":\"late\"}"
    redaction_totals=[0,0,0]
    for original in tickets.text.astype(str):
        for j,(pat,repl) in enumerate(patterns):
            original,n=pat.subn(repl,original); redaction_totals[j]+=n
        assert not any(p.search(original) for p,_ in patterns)
    def redact(t):
        counts=[]
        for pat,repl in patterns:
            t,n=pat.subn(repl,t); counts.append(n)
        assert not any(p.search(t) for p,_ in patterns)
        return t,counts
    def run(prompt,budget=None):
        mock_llm.reset(); total=0; results=[]; errs=0
        for _,r in tickets.iterrows():
            text=str(r.text); status=""; cat=""; used=0
            if not text.strip(): status="skipped_empty"
            elif mock_llm.est_tokens(text)>400: status="flagged_long"
            else:
                clean,counts=redact(text)
                # Ensure prompt never contains raw PII.
                assert not any(p.search(clean) for p,_ in patterns)
                if budget is not None and total>=budget: status="budget_exhausted"
                else:
                    for retry in range(4):
                        try:
                            resp=mock_llm.chat([{"role":"system","content":prompt},{"role":"user","content":clean}])
                            break
                        except mock_llm.TransientLLMError:
                            errs+=1
                            if retry>=3: raise
                            wait=.05*(2**retry)*random.uniform(.9,1.15)
                            print(f"retry ticket={r.ticket_id} retry={retry} wait={wait:.4f}")
                            time.sleep(wait)
                    total+=resp.usage.total_tokens; used=resp.usage.total_tokens
                    out=resp.choices[0].message.content or ""
                    try:
                        obj=json.loads(out); candidate=obj.get("category","")
                        if candidate in ["billing","delivery","technical","fraud"]:
                            cat=candidate; status="classified_json"
                        else: status="flagged_invalid_output"
                    except Exception:
                        words=re.findall(r"[a-z]+",out.lower())
                        found=[w for w in set(words) if w in ["billing","delivery","technical","fraud"]]
                        if len(found)==1: cat=found[0]; status="classified_keyword"
                        else: status="flagged_invalid_output"
            results.append({"ticket_id":r.ticket_id,"customer_id":r.customer_id,"status":status,"category":cat,
              "true_category":r.true_category,"tokens_used":used,"redacted_text":""})
        return pd.DataFrame(results),total,errs
    z,zt,ze=run(zero)
    f,ft,fe=run(few)
    classified=z[z.status.isin(["classified_json","classified_keyword"])]
    cm=pd.crosstab(classified.true_category,classified.category).to_dict()
    za=accuracy_score(classified.true_category,classified.category)
    fclass=f[f.status.isin(["classified_json","classified_keyword"])]
    fa=accuracy_score(fclass.true_category,fclass.category)
    # budget run
    b,bt,be=run(zero,6000)
    b.to_csv("outputs/triage_budget_demo.csv",index=False)
    # final routing
    out=f.copy().merge(seg[["customer_id","segment_name","is_premium"]],on="customer_id",how="left")
    out["segment_name"]=out.segment_name.fillna("")
    def route(r):
        if r.status in ["skipped_empty","flagged_long","flagged_invalid_output","budget_exhausted"]: return ("NORMAL","human_review")
        if r.category=="fraud": return ("HIGH","security_team")
        if bool(r.is_premium): return ("HIGH","premium_desk")
        return ("NORMAL","support_queue")
    vals=out.apply(route,axis=1,result_type="expand"); out["priority"]=vals[0];out["route"]=vals[1]
    out[["ticket_id","customer_id","status","category","segment_name","priority","route","tokens_used"]].to_csv("outputs/triage_results.csv",index=False)
    report={"redaction_totals":{"email":redaction_totals[0],"phone":redaction_totals[1],"order_id":redaction_totals[2],"total":sum(redaction_totals)},
      "zero_shot_classified":len(classified),"zero_shot_accuracy":za,"zero_shot_tokens":zt,"zero_shot_errors":ze,
      "few_shot_classified":len(fclass),"few_shot_accuracy":fa,"few_shot_tokens":ft,"few_shot_errors":fe,
      "zero_shot_status":z.status.value_counts().to_dict(),"few_shot_status":f.status.value_counts().to_dict(),
      "budget_exhausted":int((b.status=="budget_exhausted").sum()),"budget":6000,"zero_shot_confusion":cm}
    Path("outputs/part_c_report.json").write_text(json.dumps(report,indent=2,default=str))
    print("PART C",json.dumps(report,default=str))
    return report

def part_d():
    # chunk documents
    def chunks(overlap=10):
        docs=[]
        for fn in sorted(Path("data/policies").glob("*.txt")):
            words=fn.read_text().split()
            step=40-overlap
            for start in range(0,len(words),step):
                chunk=words[start:start+40]
                if not chunk: break
                docs.append({"source":fn.name,"text":" ".join(chunk)})
                if start+40>=len(words): break
        return docs
    docs=chunks(10); vector=TfidfVectorizer(); mat=vector.fit_transform([d["text"] for d in docs])
    def retrieve(q,k=3,ds=docs,v=vector,m=mat):
        scores=cosine_similarity(v.transform([q]),m).ravel()
        return [{"source":ds[i]["source"],"score":float(scores[i]),"text":ds[i]["text"]} for i in np.argsort(scores)[::-1][:k]]
    def answer(q,ds=docs,v=vector,m=mat):
        hits=retrieve(q,3,ds,v,m)
        context="\n".join(f"[{h['source']}] {h['text']}" for h in hits)
        res=mock_llm.chat([{"role":"system","content":"POLICY ASSISTANT: Answer only using supplied context. If absent, say I could not find the information."},
          {"role":"user","content":f"CONTEXT:\n{context}\nQUESTION:\n{q}"}])
        return res.choices[0].message.content,[h["source"] for h in hits]
    ev=pd.read_csv("data/raw/policy_eval.csv")
    good_doc=good_fact=unans=0
    for _,r in ev.iterrows():
        a,src=answer(r.question)
        if r.expected_doc!="NONE":
            good_doc+=r.expected_doc in src; good_fact+=str(r.expected_fact) in a
        else: unans+=a.startswith("I could not find")
    # no overlap evaluation
    d0=chunks(0); v0=TfidfVectorizer(); m0=v0.fit_transform([d["text"] for d in d0])
    no_doc=no_fact=no_unans=0
    no_overlap_breaks=[]
    for _,r in ev.iterrows():
        a,src=answer(r.question,d0,v0,m0)
        if r.expected_doc!="NONE":
            no_doc+=r.expected_doc in src; no_fact+=str(r.expected_fact) in a
            if str(r.expected_fact) not in a: no_overlap_breaks.append(str(r.question))
        else: no_unans+=a.startswith("I could not find")
    # tools
    Path("outputs/refund_ledger.jsonl").write_text("")
    def get_order(order_id):
        with sqlite3.connect("data/instamart.db") as con:
            row=con.execute("SELECT order_id,status,order_amount,delivery_minutes FROM orders WHERE order_id=?",(order_id,)).fetchone()
        return {"error":"order not found"} if row is None else dict(zip(["order_id","status","order_amount","delivery_minutes"],row))
    def search_policy(query): return {"results":retrieve(query)}
    def calculate_refund(order_amount,delivery_minutes):
        return {"refund_inr":round(float(order_amount),2) if float(delivery_minutes)>45 else 0.0}
    def ledger(name,order_id,refund_inr):
        with open("outputs/refund_ledger.jsonl","a") as f: f.write(json.dumps({"timestamp":datetime.now().isoformat(),"tool":name,"order_id":order_id,"refund_inr":refund_inr})+"\n")
        return {"refund_inr":refund_inr}
    def issue_refund(order_id,refund_inr): return ledger("issue_refund",order_id,refund_inr)
    def escalate_to_human(order_id,refund_inr): return ledger("escalate_to_human",order_id,refund_inr)
    TOOL_MAP={"get_order":get_order,"search_policy":search_policy,"calculate_refund":calculate_refund,"issue_refund":issue_refund,"escalate_to_human":escalate_to_human}
    TOOLS_SPEC=[{"name":"get_order","description":"Fetch an order from SQLite","parameters":{"order_id":"string"}},
      {"name":"search_policy","description":"Retrieve relevant policy chunks","parameters":{"query":"string"}},
      {"name":"calculate_refund","description":"Calculate refund based on delivery time","parameters":{"order_amount":"number","delivery_minutes":"number"}},
      {"name":"issue_refund","description":"Record an automatic refund","parameters":{"order_id":"string","refund_inr":"number"}},
      {"name":"escalate_to_human","description":"Send high-value refund to human lead","parameters":{"order_id":"string","refund_inr":"number"}}]
    # ReAct loop, tool observations serialized as required by mock.
    def run_agent(user_message,scenario_id,max_iterations=6,token_limit=6000,time_limit_seconds=10,run_id=None,crash_after=None,resume_state=None):
        started=time.monotonic()
        messages=resume_state["messages"] if resume_state else [{"role":"system","content":"AGENT: Use tools to process refund safely."},{"role":"user","content":user_message}]
        steps=resume_state["steps"] if resume_state else []
        tokens=resume_state["tokens_used"] if resume_state else 0
        executed=resume_state["tools_executed"] if resume_state else []
        calls=resume_state["llm_calls"] if resume_state else 0
        for it in range(max_iterations-calls):
            if time.monotonic()-started>=time_limit_seconds:
                return {"status":"FALLBACK_TIMEOUT","final_answer":"FALLBACK_TIMEOUT: hand off to human support.","steps":steps,"llm_calls":calls,"tokens_used":tokens}
            resp=mock_llm.chat(messages,tools=TOOLS_SPEC); calls+=1; tokens+=resp.usage.total_tokens
            if tokens>token_limit:
                return {"status":"FALLBACK_TOKEN_LIMIT","final_answer":"FALLBACK_TOKEN_LIMIT: hand off to human support.","steps":steps,"llm_calls":calls,"tokens_used":tokens}
            msg=resp.choices[0].message
            if not msg.tool_calls:
                final=msg.content or "DENIED: unable to process."
                return {"status":final.split(":")[0],"final_answer":final,"steps":steps,"llm_calls":calls,"tokens_used":tokens}
            tcs=[{"id":tc.id,"name":tc.name,"arguments":tc.arguments} for tc in msg.tool_calls]
            messages.append({"role":"assistant","content":None,"tool_calls":tcs})
            for tc in tcs:
                obs=TOOL_MAP[tc["name"]](**tc["arguments"])
                executed.append(tc["name"]); steps.append({"tool":tc["name"],"arguments":tc["arguments"],"observation":obs})
                messages.append({"role":"tool","tool_call_id":tc["id"],"name":tc["name"],"content":json.dumps(obs)})
                if run_id:
                    Path(f"state/{run_id}.json").write_text(json.dumps({"run_id":run_id,"scenario_id":scenario_id,"messages":messages,"tokens_used":tokens,"step_number":len(steps),"tools_executed":executed,"steps":steps,"llm_calls":calls},default=str))
                if crash_after and len(steps)==crash_after: raise RuntimeError("SIMULATED_CRASH")
        return {"status":"FALLBACK_MAX_ITERATIONS","final_answer":"FALLBACK_MAX_ITERATIONS: hand off to human support.","steps":steps,"llm_calls":calls,"tokens_used":tokens}
    scenarios=pd.read_csv("data/raw/agent_scenarios.csv")
    runs=[]
    for _,r in scenarios.iterrows():
        result=run_agent(r.message,r.scenario_id)
        result.update({"scenario_id":r.scenario_id})
        runs.append(result)
    with open("outputs/agent_runs.jsonl","w") as f:
        for r in runs: f.write(json.dumps(r,default=str)+"\n")
    def resume_agent(run_id):
        saved=json.loads(Path(f"state/{run_id}.json").read_text())
        return run_agent("",saved["scenario_id"],run_id=run_id,resume_state=saved)
    # demos
    s1=scenarios.iloc[0]
    token_demo=run_agent(s1.message,"S1-token",token_limit=200)
    timeout_demo=run_agent(s1.message,"S1-timeout",time_limit_seconds=0)
    # crash/resume demo with simulated crash after second observation
    rid="S1-resume"
    try: run_agent(s1.message,"S1",run_id=rid,crash_after=2)
    except RuntimeError: pass
    state=json.loads(Path(f"state/{rid}.json").read_text())
    resumed=resume_agent(rid)
    report={"chunk_count":len(docs),"chunks_by_source":pd.Series([d["source"] for d in docs]).value_counts().to_dict(),
      "rag_eval":{"doc":f"{good_doc}/12","fact":f"{good_fact}/12","fallback":f"{unans}/2"},
      "zero_overlap_eval":{"doc":f"{no_doc}/12","fact":f"{no_fact}/12","fallback":f"{no_unans}/2","break_questions":no_overlap_breaks},
      "scenarios":[{"scenario_id":r["scenario_id"],"status":r["status"],"final_answer":r["final_answer"],"llm_calls":r["llm_calls"],"tokens_used":r["tokens_used"]} for r in runs],
      "token_demo":token_demo["status"],"timeout_demo":timeout_demo["status"],
      "resume_tools":state["tools_executed"]+resumed["steps"][-2:][0:0] if False else state["tools_executed"]+[x["tool"] for x in resumed["steps"][len(state["steps"]):]],
      "resume_final":resumed["final_answer"]}
    Path("outputs/part_d_report.json").write_text(json.dumps(report,indent=2,default=str))
    print("PART D",json.dumps(report,default=str))
    return report

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--part",default="all",choices=["all","a","b","c","d"]); args=ap.parse_args()
    if not Path("data/raw/orders.csv").exists(): os.system("python generate_data.py")
    if args.part in ["all","a"]:
        c,o,p,a=part_a()
    else:
        c=pd.read_csv("data/clean/customers.csv",parse_dates=["signup_date"]); o=pd.read_csv("data/clean/orders.csv",parse_dates=["order_ts"]); p=pd.read_csv("data/clean/products.csv")
    if args.part in ["all","b"]: rfm,br=part_b(c,o,p)
    if args.part in ["all","c"]: cr=part_c()
    if args.part in ["all","d"]: dr=part_d()
if __name__=="__main__": main()
