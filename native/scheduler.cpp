#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <queue>
namespace py = pybind11;

using namespace std;

int add(int a, int b){
    return a+b;
}

double test(vector<tuple<int, double, vector<int>>> data){
    double total = 0;
    for (const auto& item : data){
        int a = get<0>(item);
        double b = get<1>(item);
        vector<int> c = get<2>(item);
        total += b;
    }
    return total;
}

vector<int> happensBeforeForOperations(vector<int> operationSequence, vector<vector<int>> all_components){
    if (operationSequence.size() != all_components.size()){
        throw invalid_argument("The size of operationSequence and all_components must be the same.");
    }
    unordered_map<int, vector<int>> happens_before_map;
    unordered_map<int, int> indegree_map;
    for (const auto& op : operationSequence){
        happens_before_map[op] = {};
        indegree_map[op] = 0;
    }
    unordered_map<int, vector<int>> ops_by_component_map;
    for (int i=0; i<all_components.size(); i++){
        for (const auto& component : all_components[i]){
            if (ops_by_component_map.find(component) == ops_by_component_map.end()){
                ops_by_component_map[component] = {};
            }
        }
    }

    for (int i=0; i<operationSequence.size(); i++){
        int op = operationSequence[i];
        vector<int> components = all_components[i];
        set<int> unique_components(components.begin(), components.end());
        for (const auto& component : unique_components){
            for (auto& prev_op : ops_by_component_map[component]){
                if (find(happens_before_map[prev_op].begin(), happens_before_map[prev_op].end(), op) == happens_before_map[prev_op].end()){
                    happens_before_map[prev_op].push_back(op);
                    indegree_map[op]++;
                }
            }
            ops_by_component_map[component].push_back(op);
        }
    }

    queue<int> zero_indegree_queue;
    for (const auto& op : operationSequence){
        if (indegree_map[op] == 0){
            zero_indegree_queue.push(op);
        }
    }
    vector<int> topologically_sorted_ops;
    while (!zero_indegree_queue.empty()){
        int op = zero_indegree_queue.front();
        zero_indegree_queue.pop();
        topologically_sorted_ops.push_back(op);
        for (const auto& neighbour : happens_before_map[op]){
            indegree_map[neighbour]--;
            if (indegree_map[neighbour] == 0){
                zero_indegree_queue.push(neighbour);
            }
        }
    }

    return topologically_sorted_ops;
}

PYBIND11_MODULE(scheduler, m) {
    m.def("add", &add, "A function that adds two numbers");
    m.def("test", &test, "A function that processes a vector of tuples");
    m.def("happensBeforeForOperations", &happensBeforeForOperations, "A function that determines the order of operations based on dependencies");
}