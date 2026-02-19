"""
DSA Topics Definition
========================
Defines the 16-week DSA curriculum with topic details,
subtopics, key concepts, and difficulty progression.

Curriculum Learning Strategy:
─────────────────────────────
Topics are ordered from foundational to advanced:
  1. Arrays & Strings (Week 1-2) - Foundation
  2. Linked Lists (Week 3-4) - Pointer manipulation
  3. Stacks & Queues (Week 5-6) - Abstract data types
  4. Trees (Week 7-8) - Hierarchical structures
  5. Graphs (Week 9-10) - General structures
  6. Sorting & Searching (Week 11-12) - Algorithms
  7. Dynamic Programming (Week 13-14) - Optimization
  8. Advanced / Review (Week 15-16) - Integration

This progressive ordering ensures that:
- Earlier topics provide foundation for later ones
- Model builds understanding incrementally
- Forgetting of earlier topics is monitored
"""

from typing import Dict, List, Any


DSA_TOPICS: Dict[str, Dict[str, Any]] = {
    
    # ══════════════════════════════════════════════════════════
    # WEEK 1-2: Arrays & Strings
    # ══════════════════════════════════════════════════════════
    "arrays_strings": {
        "name": "Arrays and Strings",
        "weeks": [1, 2],
        "description": (
            "Arrays are contiguous memory structures supporting O(1) random access. "
            "Strings are character arrays with specialized operations. "
            "Key techniques include two-pointer, sliding window, prefix sums, "
            "and hash-based approaches."
        ),
        "subtopics": [
            "Array traversal and manipulation",
            "Two-pointer technique",
            "Sliding window technique",
            "Prefix sum arrays",
            "String matching and manipulation",
            "Hash map for frequency counting",
            "In-place array operations",
            "Subarray problems (max subarray, subarray sum)",
            "Matrix operations (2D arrays)",
            "Anagram and palindrome problems",
        ],
        "key_concepts": [
            "contiguous memory", "random access", "O(1) index",
            "two pointer", "sliding window", "prefix sum",
            "in-place", "subarray", "hash map", "anagram",
        ],
        "difficulty_progression": {
            "easy": ["array traversal", "find max/min", "reverse array"],
            "medium": ["two sum", "sliding window max", "subarray sum equals k"],
            "hard": ["trapping rain water", "median of two sorted arrays"],
        },
        "prerequisite_topics": [],
        "sample_questions": [
            "Given an array of integers, find two numbers that add up to a target. What is the optimal time complexity?",
            "Implement the sliding window technique to find the maximum sum subarray of size k.",
            "Explain the two-pointer technique and when it is applicable.",
            "Write a function to check if two strings are anagrams of each other.",
        ],
    },
    
    # ══════════════════════════════════════════════════════════
    # WEEK 3-4: Linked Lists
    # ══════════════════════════════════════════════════════════
    "linked_lists": {
        "name": "Linked Lists",
        "weeks": [3, 4],
        "description": (
            "Linked lists are dynamic data structures where elements are connected "
            "via pointers. They support O(1) insertion/deletion at known positions "
            "but O(n) access. Key techniques include fast/slow pointers, "
            "dummy nodes, and recursive approaches."
        ),
        "subtopics": [
            "Singly linked list operations",
            "Doubly linked list operations",
            "Circular linked list",
            "Fast and slow pointer technique",
            "Reversing a linked list",
            "Merging sorted linked lists",
            "Cycle detection (Floyd's algorithm)",
            "Finding middle element",
            "Intersection of two linked lists",
            "LRU Cache using linked list + hash map",
        ],
        "key_concepts": [
            "node", "pointer", "head", "tail", "singly", "doubly",
            "circular", "fast slow pointer", "reverse", "cycle detection",
        ],
        "difficulty_progression": {
            "easy": ["insert at head", "traverse", "delete node"],
            "medium": ["reverse list", "detect cycle", "merge two lists"],
            "hard": ["reverse k-group", "copy list with random pointer"],
        },
        "prerequisite_topics": ["arrays_strings"],
        "sample_questions": [
            "Explain Floyd's cycle detection algorithm with its mathematical proof.",
            "Implement a function to reverse a singly linked list iteratively and recursively.",
            "How would you find the intersection node of two singly linked lists?",
        ],
    },
    
    # ══════════════════════════════════════════════════════════
    # WEEK 5-6: Stacks & Queues
    # ══════════════════════════════════════════════════════════
    "stacks_queues": {
        "name": "Stacks and Queues",
        "weeks": [5, 6],
        "description": (
            "Stacks (LIFO) and Queues (FIFO) are fundamental abstract data types. "
            "Stacks support push/pop operations, queues support enqueue/dequeue. "
            "Applications include expression evaluation, BFS, and monotonic "
            "stack/queue techniques for optimization problems."
        ),
        "subtopics": [
            "Stack implementation and operations",
            "Queue implementation and operations",
            "Monotonic stack technique",
            "Monotonic queue technique",
            "Balanced parentheses",
            "Next greater/smaller element",
            "Min/Max stack",
            "Queue using two stacks",
            "Circular queue",
            "Priority queue (Heap basics)",
            "Infix to postfix conversion",
            "Expression evaluation",
        ],
        "key_concepts": [
            "LIFO", "FIFO", "push", "pop", "enqueue", "dequeue",
            "monotonic stack", "priority queue", "balanced parentheses",
        ],
        "difficulty_progression": {
            "easy": ["valid parentheses", "implement stack", "implement queue"],
            "medium": ["next greater element", "daily temperatures", "min stack"],
            "hard": ["largest rectangle in histogram", "sliding window maximum"],
        },
        "prerequisite_topics": ["arrays_strings", "linked_lists"],
        "sample_questions": [
            "Design a stack that supports push, pop, top, and retrieving the minimum element in O(1) time.",
            "Explain the monotonic stack technique with an example problem.",
            "Implement a queue using two stacks. Analyze the amortized time complexity.",
        ],
    },
    
    # ══════════════════════════════════════════════════════════
    # WEEK 7-8: Trees
    # ══════════════════════════════════════════════════════════
    "trees": {
        "name": "Trees",
        "weeks": [7, 8],
        "description": (
            "Trees are hierarchical data structures with a root node and children. "
            "Binary trees and BSTs are fundamental. Key operations include traversal "
            "(inorder, preorder, postorder, level-order), searching, insertion, "
            "deletion, and balancing (AVL, Red-Black)."
        ),
        "subtopics": [
            "Binary tree traversals (inorder, preorder, postorder)",
            "Level-order traversal (BFS on trees)",
            "Binary Search Tree (BST) operations",
            "AVL tree rotations and balancing",
            "Tree height and depth computation",
            "Lowest Common Ancestor (LCA)",
            "Tree diameter",
            "Serialize and deserialize a tree",
            "Balanced tree check",
            "Path sum problems",
            "Binary tree to doubly linked list",
        ],
        "key_concepts": [
            "root", "leaf", "height", "depth", "BST property",
            "balanced", "inorder", "preorder", "postorder", "level order",
            "rotation", "AVL", "LCA", "diameter",
        ],
        "difficulty_progression": {
            "easy": ["tree traversal", "max depth", "is balanced"],
            "medium": ["LCA", "validate BST", "tree diameter"],
            "hard": ["serialize tree", "recover BST", "binary tree maximum path sum"],
        },
        "prerequisite_topics": ["arrays_strings", "linked_lists", "stacks_queues"],
        "sample_questions": [
            "Given a binary tree, find the lowest common ancestor of two given nodes.",
            "Explain AVL tree rotations (LL, RR, LR, RL) with diagrams.",
            "Write code to serialize and deserialize a binary tree.",
        ],
    },
    
    # ══════════════════════════════════════════════════════════
    # WEEK 9-10: Graphs
    # ══════════════════════════════════════════════════════════
    "graphs": {
        "name": "Graphs",
        "weeks": [9, 10],
        "description": (
            "Graphs are general structures with vertices and edges. They can be "
            "directed/undirected, weighted/unweighted. Key algorithms include BFS, "
            "DFS, shortest paths (Dijkstra, Bellman-Ford), minimum spanning trees "
            "(Kruskal, Prim), and topological sorting."
        ),
        "subtopics": [
            "Graph representations (adjacency list/matrix)",
            "BFS (Breadth-First Search)",
            "DFS (Depth-First Search)",
            "Topological sorting",
            "Dijkstra's shortest path",
            "Bellman-Ford algorithm",
            "Floyd-Warshall all-pairs shortest path",
            "Kruskal's MST",
            "Prim's MST",
            "Union-Find (Disjoint Set Union)",
            "Cycle detection (directed and undirected)",
            "Bipartite graph check",
            "Connected components",
        ],
        "key_concepts": [
            "vertex", "edge", "directed", "undirected", "weighted",
            "adjacency list", "BFS", "DFS", "shortest path", "MST",
            "topological sort", "union find", "cycle detection",
        ],
        "difficulty_progression": {
            "easy": ["BFS traversal", "DFS traversal", "connected components"],
            "medium": ["topological sort", "dijkstra", "cycle detection"],
            "hard": ["network flow", "articulation points", "strongly connected components"],
        },
        "prerequisite_topics": ["arrays_strings", "stacks_queues", "trees"],
        "sample_questions": [
            "Implement Dijkstra's algorithm and explain why it fails with negative weights.",
            "Compare BFS and DFS: when would you use each? What are their time complexities?",
            "Explain the Union-Find data structure with path compression and union by rank.",
        ],
    },
    
    # ══════════════════════════════════════════════════════════
    # WEEK 11-12: Sorting & Searching
    # ══════════════════════════════════════════════════════════
    "sorting_searching": {
        "name": "Sorting and Searching",
        "weeks": [11, 12],
        "description": (
            "Sorting arranges elements in order; searching finds elements efficiently. "
            "Comparison-based sorts have Ω(n log n) lower bound. Non-comparison sorts "
            "can achieve O(n). Binary search variants are widely applicable."
        ),
        "subtopics": [
            "Comparison-based sorting (merge, quick, heap)",
            "Non-comparison sorting (counting, radix, bucket)",
            "In-place vs stable sorting",
            "Quick sort partitioning strategies",
            "Merge sort for linked lists",
            "Binary search and its variants",
            "Search in rotated sorted array",
            "Kth largest/smallest element",
            "Order statistics",
            "External sorting",
        ],
        "key_concepts": [
            "comparison sort", "stable sort", "in-place", "divide and conquer",
            "partition", "merge", "binary search", "lower bound",
            "O(n log n)", "O(n²)", "kth element",
        ],
        "difficulty_progression": {
            "easy": ["binary search", "insertion sort", "selection sort"],
            "medium": ["merge sort", "quick sort", "search rotated array"],
            "hard": ["median of two arrays", "count inversions", "external sorting"],
        },
        "prerequisite_topics": ["arrays_strings"],
        "sample_questions": [
            "Prove that comparison-based sorting has an Ω(n log n) lower bound.",
            "Compare merge sort and quick sort in terms of time complexity, space, and stability.",
            "Implement binary search to find the first occurrence of an element in a sorted array.",
        ],
    },
    
    # ══════════════════════════════════════════════════════════
    # WEEK 13-14: Dynamic Programming
    # ══════════════════════════════════════════════════════════
    "dynamic_programming": {
        "name": "Dynamic Programming",
        "weeks": [13, 14],
        "description": (
            "Dynamic Programming solves optimization problems by breaking them into "
            "overlapping subproblems with optimal substructure. Two approaches: "
            "top-down (memoization) and bottom-up (tabulation). Requires identifying "
            "state, transitions, and base cases."
        ),
        "subtopics": [
            "Memoization (top-down DP)",
            "Tabulation (bottom-up DP)",
            "Fibonacci and number theory DP",
            "0/1 Knapsack and variants",
            "Longest Common Subsequence (LCS)",
            "Longest Increasing Subsequence (LIS)",
            "Edit distance (Levenshtein)",
            "Coin change problem",
            "Matrix chain multiplication",
            "DP on trees",
            "Bitmask DP",
            "State space reduction",
        ],
        "key_concepts": [
            "optimal substructure", "overlapping subproblems",
            "memoization", "tabulation", "state", "transition",
            "base case", "recurrence relation", "bottom-up", "top-down",
        ],
        "difficulty_progression": {
            "easy": ["fibonacci", "climbing stairs", "coin change (basic)"],
            "medium": ["LCS", "knapsack", "edit distance"],
            "hard": ["burst balloons", "palindrome partitioning", "bitmask DP"],
        },
        "prerequisite_topics": ["arrays_strings", "sorting_searching"],
        "sample_questions": [
            "Explain the difference between memoization and tabulation with examples.",
            "Solve the 0/1 Knapsack problem. Define the state, transition, and base case.",
            "Write the recurrence relation for the Longest Common Subsequence problem.",
        ],
    },
    
    # ══════════════════════════════════════════════════════════
    # WEEK 15: Advanced Topics
    # ══════════════════════════════════════════════════════════
    "advanced_topics": {
        "name": "Advanced Topics",
        "weeks": [15],
        "description": (
            "Advanced topics combining multiple data structures and techniques. "
            "Includes tries, segment trees, advanced graph algorithms, "
            "and problem patterns that require combining earlier topics."
        ),
        "subtopics": [
            "Trie (prefix tree)",
            "Segment tree",
            "Binary Indexed Tree (BIT/Fenwick Tree)",
            "Advanced graph: network flow",
            "String algorithms (KMP, Rabin-Karp)",
            "Greedy algorithms",
            "Backtracking",
            "Bit manipulation",
        ],
        "key_concepts": [
            "trie", "segment tree", "fenwick tree", "KMP",
            "greedy", "backtracking", "bit manipulation",
        ],
        "difficulty_progression": {
            "easy": ["implement trie", "basic greedy"],
            "medium": ["range query with segment tree", "KMP pattern matching"],
            "hard": ["max flow min cut", "suffix array"],
        },
        "prerequisite_topics": ["trees", "graphs", "dynamic_programming"],
        "sample_questions": [
            "Implement a trie and explain how it supports prefix-based search.",
            "Explain the KMP string matching algorithm with failure function.",
        ],
    },
    
    # ══════════════════════════════════════════════════════════
    # WEEK 16: Comprehensive Review
    # ══════════════════════════════════════════════════════════
    "comprehensive_review": {
        "name": "Comprehensive Review",
        "weeks": [16],
        "description": (
            "Final review covering all topics. Mixed-topic questions that require "
            "combining multiple data structures and algorithms. Focus on problem "
            "selection and choosing the right approach."
        ),
        "subtopics": [
            "Mixed topic problems",
            "System design (basic)",
            "Problem-solving strategies",
            "Complexity analysis mastery",
            "Interview-style questions",
        ],
        "key_concepts": [
            "problem decomposition", "algorithm selection",
            "trade-offs", "optimization", "system design",
        ],
        "difficulty_progression": {
            "easy": ["identify the right data structure"],
            "medium": ["combine two techniques"],
            "hard": ["open-ended design problems"],
        },
        "prerequisite_topics": ["all"],
        "sample_questions": [
            "Design a system to find the top K frequent elements from a stream of data.",
            "Given a problem, explain how to choose between BFS, DFS, DP, and Greedy.",
        ],
    },
}


def get_topic_names() -> List[str]:
    """Return all topic keys in curriculum order."""
    return [
        "arrays_strings", "linked_lists", "stacks_queues",
        "trees", "graphs", "sorting_searching",
        "dynamic_programming", "advanced_topics", "comprehensive_review",
    ]


def get_topic_by_week(week: int) -> str:
    """Get the topic key for a given week number."""
    for topic_key, topic_info in DSA_TOPICS.items():
        if week in topic_info.get("weeks", []):
            return topic_key
    return "comprehensive_review"


def get_all_sample_questions() -> Dict[str, List[str]]:
    """Get all sample questions organized by topic."""
    return {
        topic: info.get("sample_questions", [])
        for topic, info in DSA_TOPICS.items()
    }
