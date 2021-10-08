import sys
from collections import OrderedDict, defaultdict, deque
from time import process_time

sys.setrecursionlimit(10**6)

starttime = process_time()


class Node:
    def __init__(self, author_name, frequency, parent_node):
        self.author_name = author_name
        self.frequency = frequency
        self.parent = parent_node
        self.children = {}
        self.next = None

    def increase(self, frequency):
        self.frequency += frequency


def parseDatasetFromFile(file_name):
    dataset = []
    frequency = []

    f = open(file_name, "r")

    while True:
        line = f.readline()

        if not line:
            break
        line = line.replace("\n", "")
        line_data = line.split('#')
        dataset.append(line_data)
        frequency.append(1)
    f.close()

    return dataset, frequency


# Reads dataset from file and performs the FP growth algorithm on it
def FPGrowthFromFile(file_name, threshold):

    print("parsing...")
    dataset, frequency = parseDatasetFromFile(file_name)

    print("building tree...")
    fp_tree, header_table = buildTree(dataset, frequency, threshold)

    print("finding max...")
    frequent_items = []
    findMaxOnEachLayer(fp_tree)
    # mineTree(header_table, threshold, set(), frequent_items)
    # rules = associationRule(frequent_items, )  # Do we need this?
    # return frequent_items


def buildTree(dataset, frequency, threshold):
    # Like a normal dictionary, but doesn't raise key errors
    header_table = defaultdict(int)

    # Count frequency and init header table
    for i, authorlist in enumerate(dataset):
        for author in authorlist:
            header_table[author] += frequency[i]

    # Filter the frequent items
    header_table = dict((author, support) for author, support in header_table.items() if support > threshold)

    # We found no frequent sets in this case
    if len(header_table) == 0:
        return None, None

    # Column in header table: [author: [frequency, parent]]
    for author in header_table:
        header_table[author] = [header_table[author], None]

    # Init tree root
    fp_tree = Node('Null', 1, None)

    # Update FP tree for each sorted author list consisting of frequent items
    for i, authorlist in enumerate(dataset):
        authorlist = [author for author in authorlist if author in header_table]
        authorlist.sort(key=lambda author: header_table[author][0], reverse=True)

        current_node = fp_tree

        # For each author, insert it into the tree by following the path to a leave and
        # appending it as a child (if it didn't exist yet)
        for author in authorlist:
            current_node = updateTree(author, current_node, header_table, frequency[i])

    #print("Time to build FP tree : {0} s".format(process_time() - starttime))
    return fp_tree, header_table


def updateTree(author, node, header_table, frequency):
    # We only need to increase the count for this author if it already exists as a child of the current node
    if author in node.children:
        node.children[author].increase(frequency)

    else:
        # Otherwise we create a new leaf for this author and append it to its parent
        new_author_node = Node(author, frequency, node)
        node.children[author] = new_author_node
        updateHeaderTable(author, new_author_node,header_table)

    # We return the leaf of the author we were evaluating, so we can continue from that leaf (all the way down the list)
    return node.children[author]


def updateHeaderTable(author, new_parent_node, header_table):
    # We want to update the header table, so it contains the same information as our tree
    # Therefore, we let the table know who the parent of the author that we just appended to the tree is.
    if header_table[author][1] is None:
        header_table[author][1] = new_parent_node
    else:
        current_node = header_table[author][1]

        # Go down the 'tree' (table representation) and add the parent node to the author who doesn't have a parent
        while current_node.next is not None:
            current_node = current_node.next
        current_node.next = new_parent_node


def mineTree(header_table, threshold, prefix, frequent_items):
    # Sort authors by frequency and list them
    authors_sorted = [author[0] for author in sorted(list(header_table.items()), key=lambda n:n[1][0])]

    # Ascending order
    for author in authors_sorted:
        # Pattern growth achieved by concatenation of suffix pattern with frequent patterns
        # generated from conditional FP tree
        freq_set = prefix.copy()
        freq_set.add(author)
        frequent_items.append(freq_set)

        # Build the conditional pattern base by finding all prefix paths, then build the conditional FP tree
        conditional_pattern_base, frequency = findAllPrefixes(author, header_table)
        conditional_tree, new_header_table = buildTree(conditional_pattern_base, frequency, threshold)

        if new_header_table is not None:
            # Mine recursively down on this new header table formed from the FP conditional tree
            mineTree(new_header_table, threshold, freq_set, frequent_items)
    #print("Time to mine FP tree : {0} s".format(process_time() - starttime))


def findAllPrefixes(base_pattern, header_table):
    # First node in linked list
    node = header_table[base_pattern][1]

    conditional_patterns = []
    frequency = []

    while node is not None:
        prefix_path = []
        # Go from the base pattern node all the way up to the root (on the way, we build up the prefix path)
        ascendFPTree(node, prefix_path)
        if len(prefix_path) > 1:
            conditional_patterns.append(prefix_path[1:])
            frequency.append(node.frequency)

        # Do the same for all the next nodes
        node = node.next
    return conditional_patterns, frequency


def ascendFPTree(node, prefix_path):
    # Recursively go up the tree and build the prefix path
    if node.parent is not None:
        prefix_path.append(node)
        ascendFPTree(node.parent, prefix_path)


# def checkLayerForOccurence(root, stop_layer, author_name):
#     frequency = 0
#     curr_layer = 1
#     if stop_layer == 1:
#         for key, value in root.children.items():
#             if value.author_name == author_name:
#                 frequency += value.frequency
#
#     if stop_layer == 2:
#         for key, value in root.children.items():
#             for child in value.children.values():
#                 if value.author_name == author_name:
#                     frequency += child.frequency
#     return frequency


# Perform BFS recursively on the graph
def recursiveBFS(graph, q, discovered):
    if not q:
        return

    # dequeue front node and print it
    v = q.popleft()
    print(v, end=' ')

    # do for every edge `v —> u`
    for u in graph.adjList[v]:
        if not discovered[u]:
            # mark it as discovered and enqueue it
            discovered[u] = True
            q.append(u)

    recursiveBFS(graph, q, discovered)

def findMaxOnEachLayer(root: Node):
    LIMIT = 1000
    current_layer: int = 0

    children: deque[Node] = deque()
    children.append(root)
    first_child_of_next_layer: Node = list(root.children.values())[0]

    items_on_layer = 0
    inserted = 0
    max_curr_layer: int = 0
    max_freq_sets: list[str] = []

    while children and current_layer < LIMIT:
        node_to_process = children.popleft()

        if node_to_process is first_child_of_next_layer:
            print(current_layer, items_on_layer, inserted)

            if current_layer != 0:
                prefix: list[Node] = []
                ascendFPTree(max_freq_sets[0], prefix)

                group = []
                for a in prefix:
                    group.append(a.author_name)

                print("> {0} {1}".format(max_curr_layer, group))
            
            inserted = 0
            items_on_layer = 0
            current_layer += 1

            max_curr_layer = 0
            max_freq_sets = []

        items_on_layer += 1

        # OK TODO: Do something with node
        if node_to_process.frequency > max_curr_layer:
            max_curr_layer = node_to_process.frequency
            max_freq_sets = []
        
        if node_to_process.frequency == max_curr_layer:
            max_freq_sets.append(node_to_process)

        for child in node_to_process.children.values():
            if inserted == 0:
                first_child_of_next_layer = list(node_to_process.children.values())[0]

            children.append(child)
            inserted += 1

freq_items = FPGrowthFromFile('../data/dblp.txt', 1)
# print(freq_items)
