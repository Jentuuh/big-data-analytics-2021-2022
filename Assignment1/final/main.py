from dataset import parse_formatted, parse_etree, dataset_to_file, parse


# parse_etree()


# dataset = parse()
dataset = parse_formatted()
# dataset_to_file(dataset, "../../data/formatted.dblp.txt")


# mx = 0
# baskets = 0
# total_items = 0
# gt_1000 = 0 # 1000:2, 100:45, 50:295, 25:1986, 20:3714


# print(dataset[0:35])

# for data in dataset:
#   total_items += len(data)

#   if len(data) > 0:
#     baskets += 1

#   if len(data) > 100:
#     gt_1000 += 1
#     # print(len(data))

#   if len(data) > mx:
#     mx = len(data)

# print(mx, baskets, total_items, gt_1000)