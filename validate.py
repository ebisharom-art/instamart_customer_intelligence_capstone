import json
from pathlib import Path
import pandas as pd
a=json.loads(Path("outputs/part_a_audit.json").read_text())
b=json.loads(Path("outputs/part_b_report.json").read_text())
c=json.loads(Path("outputs/part_c_report.json").read_text())
r=pd.read_csv("data/clean/orders.csv")
assert len(r)==12309
assert (r.status=="completed").sum()==11110
assert (r.status=="cancelled").sum()==1199
assert round(r.loc[r.status=="completed","order_amount"].sum(),2)==2998668.80
assert a["orders_exact_duplicates"]==119
assert a["orders_pending"]==656
assert a["non_numeric_amount"]==120
assert a["negative_delivery"]==106
assert b["rfm_customers"]==1175
assert b["cohort"]==1156 and b["lapsed"]==443
assert c["zero_shot_classified"]==147 and c["few_shot_classified"]==147
assert c["zero_shot_accuracy"]>0.85 and c["few_shot_accuracy"]==1.0
print("ALL CHECKPOINTS PASSED")
