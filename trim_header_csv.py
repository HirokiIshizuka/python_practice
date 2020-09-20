# CSVファイルを整形する
in_file = "data.csv"
out_file = "kion15y.csv"

with open(in_file, "rt", encoding="Shift_JIS") as fr:
    lines = fr.readlines()

lines = ["年,月,日,気温,品質,均質\n"] + lines[5:]
lines = map(lambda v: v.replace('/', ','), lines)
result = "".join(lines).strip()
print(result)

with open(out_file, "wt", encoding="utf-8") as fw:
    fw.write(result)
    print("saved")