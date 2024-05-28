#include<torch/extension.h>
#include<iostream>
#include<sstream>
#include<string>
#include<map>
#include<vector>
#include<tuple>
#include<deque>
#include<utility>
#include<boost/graph/adjacency_list.hpp>

class tree_node{
    public:
        int label, idx;
        std::vector<std::pair<int, int>> children;
        tree_node(const int &lab, const int &idx_){
            label = lab;
            idx = idx_;
        }
        void insert_child(const int &child_idx, const int &edge_label){
            children.push_back(std::make_pair(child_idx, edge_label));
        }
};


typedef boost::adjacency_list<boost::vecS, boost::vecS, boost::undirectedS,
        boost::no_property, boost::no_property> BGraph;


bool bin_pred(std::string s1, std::string s2){
    std::istringstream ss1(s1), ss2(s2);
    std::string tmp1, tmp2;
    int l1, l2;
    while(ss1 >> tmp1 && ss2 >> tmp2){
        if(tmp1.compare("$") == 0){
            return false;
        }
        if(tmp2.compare("$") == 0)
            return true;
        if(tmp1.compare(tmp2) == 0){
            continue;
        }
        l1 = std::stoi(tmp1);
        l2 = std::stoi(tmp2);
        return l1 < l2;
    }
    if(ss2)
        return false;
    if(ss1)
        return true;
}


std::string get_can_lab_tree(const std::vector<tree_node*> &tree, int root=0){
    std::string s, child_label;
    std::stringstream ss;
    ss << tree[root]->label;
    if (tree[root]->children.size() != 0){
        std::vector<std::string> child_labels;
        for(std::vector<std::pair<int, int>>::iterator it=tree[root]->children.begin(); it!=tree[root]->children.end(); it++){
            child_label = get_can_lab_tree(tree, it->first);
            child_label.insert(0, 1, ' ');
            child_label.insert(0, std::to_string(it->second));
            child_labels.push_back(child_label);
        }
        // std::sort(child_labels.begin(), child_labels.end(), bin_pred);
        std::sort(child_labels.begin(), child_labels.end());
        for(std::vector<std::string>::iterator it=child_labels.begin(); it!=child_labels.end(); it++){
            ss << " " << *it;
        }
    }
    ss << " $";
    std::getline(ss, s);
    return s;
}


void get_canonical_labels(const BGraph &G,
        const std::vector<int> &node_attrs,
        const std::vector<int> &edge_attrs,
        std::map<std::pair<int, int>, int> &edge_pair_to_ind,
        std::vector<std::string> &canonical_labels,
        const int &n_hops){
    /* 1. Create tree for each node.
     * 1.1 Renumber nodes while creating tree, and remap them to correct
     *     node_attr and correct edge_attr.
     * 2. Get label for each tree.
     * 3. Collect the label inside a vector, use it to extend canonical_labels.
     */
    // Create Queue
    int c_hops, node_ctr, new_node_id;
    std::deque<std::tuple<int, int, int>> Q;
    boost::graph_traits<BGraph>::vertex_iterator vbegin, vend;
    int current_node; // boost::graph_traits<BGraph>::vertex_descriptor should be the same as int

    boost::tie(vbegin, vend) = boost::vertices(G);
    for(boost::graph_traits<BGraph>::vertex_iterator it=vbegin; it!=vend; it++){
        c_hops = 0;
        node_ctr = 0;
        std::vector<tree_node*> tree;
        tree.push_back(new tree_node(node_attrs[*it], node_ctr));
        Q.push_back(std::make_tuple(*it, c_hops, node_ctr++));
        do{
            std::tie(current_node, c_hops, new_node_id) = Q.front();
            Q.pop_front();
            if(c_hops >= n_hops)
                break;
            boost::graph_traits<BGraph>::adjacency_iterator abegin, aend, jt;
            boost::tie(abegin, aend) = adjacent_vertices(current_node, G);
            for(jt=abegin; jt!=aend; jt++){
                std::pair<int, int> edge_pair = std::make_pair(current_node, *jt);
                int node_label = node_attrs[*jt];
                int edge_id = edge_pair_to_ind[edge_pair];
                int edge_label = edge_attrs[edge_id];
                tree[new_node_id]->insert_child(node_ctr, edge_label);
                tree.push_back(new tree_node(node_label, node_ctr));
                Q.push_back(std::make_tuple(*jt, c_hops + 1, node_ctr++));
            }
        }while(c_hops < n_hops && !Q.empty());
        if(!Q.empty())
        Q.clear();
        std::string can = get_can_lab_tree(tree);
        canonical_labels.push_back(can);
    }
}


std::vector<std::string> data_to_BGraph(py::object data, int n_hops){
    py::object x = data.attr("x");
    py::object edge_index = data.attr("edge_index");
    py::object edge_attr = data.attr("edge_attr");
    py::object y = data.attr("y");
    py::int_ N_ = x.attr("size")(0); // number of nodes
    py::int_ M_ = edge_attr.attr("size")(0); // number of edges
    py::int_ u_, v_;
    int N = N_;
    int M = M_;
    int u, v;
    std::map<std::pair<int, int>, int> edge_pair_to_ind;
    BGraph G(N);
    for(int i=0; i<M; i++){
        u_ = edge_index[py::cast(0)][py::cast(i)];
        v_ = edge_index[py::cast(1)][py::cast(i)];
        u = u_;
        v = v_;
        if (edge_pair_to_ind.find(std::make_pair(u,v)) != edge_pair_to_ind.end())
            continue;
        edge_pair_to_ind.insert(std::make_pair(std::make_pair(u, v), i));
        edge_pair_to_ind.insert(std::make_pair(std::make_pair(v, u), i));
        boost::add_edge(u, v, G);
    }
    std::vector<int> node_attrs;
    std::vector<int> edge_attrs;
    for(auto el : x){
        py::int_ item_ = el.attr("item")();
        int itm_ = item_;
        node_attrs.push_back(itm_);
    }
    for(auto el : edge_attr){
        py::int_ item_ = el.attr("item")();
        int itm_ = item_;
        edge_attrs.push_back(itm_);
    }
    std::vector<std::string> canonical_labels;
    get_canonical_labels(G, node_attrs, edge_attrs, edge_pair_to_ind,
            canonical_labels, n_hops);
    return canonical_labels;
}

std::vector<std::string> data_to_BGraph_ID(py::object data, int n_hops){
    py::object x = data.attr("id");
    py::object edge_index = data.attr("edge_index");
    py::object edge_attr = data.attr("edge_attr");
    py::object y = data.attr("y");
    py::int_ N_ = x.attr("size")(0); // number of nodes
    py::int_ M_ = edge_attr.attr("size")(0); // number of edges
    py::int_ u_, v_;
    int N = N_;
    int M = M_;
    int u, v;
    std::map<std::pair<int, int>, int> edge_pair_to_ind;
    BGraph G(N);
    for(int i=0; i<M; i++){
        u_ = edge_index[py::cast(0)][py::cast(i)];
        v_ = edge_index[py::cast(1)][py::cast(i)];
        u = u_;
        v = v_;
        if (edge_pair_to_ind.find(std::make_pair(u,v)) != edge_pair_to_ind.end())
            continue;
        edge_pair_to_ind.insert(std::make_pair(std::make_pair(u, v), i));
        edge_pair_to_ind.insert(std::make_pair(std::make_pair(v, u), i));
        boost::add_edge(u, v, G);
    }
    std::vector<int> node_id;
    std::vector<int> edge_attrs;
    for(auto el : x){
        py::int_ item_ = el.attr("item")();
        int itm_ = item_;
        node_id.push_back(itm_);
    }
    for(auto el : edge_attr){
        py::int_ item_ = el.attr("item")();
        int itm_ = item_;
        edge_attrs.push_back(itm_);
    }
    std::vector<std::string> canonical_labels;
    get_canonical_labels(G, node_id, edge_attrs, edge_pair_to_ind,
            canonical_labels, n_hops);
    return canonical_labels;
}

py::list canonical(py::list list, py::int_ n_hops){
    /* This function gets a list of PyTorch Geometric Data Objects that
     * contain `x`, `edge_index`, `edge_attr`, `y`. Each of which are long tensors.
     *
     * Since we'll be using pybind lists, we will no longer have direct
     * access to at::Tensor functions. But no worries, we just need to
     * extract the graph from Data objects and perform the computation
     * and return the canonical labels back to the code in a list.
     *
     * This part can then be multithreaded using <thread>.
     * Do check what pybind11 does with GIL.
     * https://octavifs.com/post/pybind11-multithreading-parallellism-python/
     */
    std::vector<std::vector<std::string>> result_vector;
    std::vector<std::string> dfs_canonical_labels;
    std::vector<std::string> id_canonical_labels;
    int n_hops_ = n_hops;
    std::cout<<"Generating DFS Codes..."<<std::endl;
    for(auto item : list){
        std::vector<std::string> canonical_labels;
        dfs_canonical_labels = data_to_BGraph(py::reinterpret_borrow<py::object>(item), n_hops_);
        id_canonical_labels = data_to_BGraph_ID(py::reinterpret_borrow<py::object>(item), n_hops);
        for(int i = 0; i < dfs_canonical_labels.size(); i++)
        {
            canonical_labels.push_back(dfs_canonical_labels[i] + "/" + id_canonical_labels[i]);
        }
        result_vector.push_back(canonical_labels);
    }
    std::cout << "Generation complete." << std::endl;
    return py::cast(result_vector);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){
    m.def("canonical",&canonical,
            "Compute canonical labels for graphs");
}