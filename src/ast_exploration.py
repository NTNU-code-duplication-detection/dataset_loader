"""
AST Exploration Script

Purpose: Understand how javalang parses Java code and what the AST structure
looks like, to inform decisions about semantic chunking strategies.

Usage:
    python src/ast_exploration.py

This script helps answer:
- What AST nodes does javalang produce?
- What could "semantic chunks" look like?
- How granular can we get?
"""

from pathlib import Path

import javalang  # pylint: disable=import-error


def _extract_node_attrs(node) -> list[str]:
    """Extract displayable attributes from an AST node."""
    attrs = []
    if hasattr(node, 'name') and node.name:
        attrs.append(f"name='{node.name}'")
    if hasattr(node, 'type') and node.type:
        type_name = node.type.name if hasattr(node.type, 'name') else str(node.type)
        attrs.append(f"type='{type_name}'")
    if hasattr(node, 'value') and node.value is not None:
        attrs.append(f"value='{node.value}'")
    if hasattr(node, 'operator') and node.operator:
        attrs.append(f"op='{node.operator}'")
    return attrs


def _process_ast_children(node, indent: int, max_depth: int) -> None:
    """Recursively process and print children of an AST node."""
    for child in node.children:
        if child is None:
            continue
        if isinstance(child, list):
            for item in child:
                print_ast_tree(item, indent + 1, max_depth)
        else:
            print_ast_tree(child, indent + 1, max_depth)


def print_ast_tree(node, indent: int = 0, max_depth: int = 10) -> None:
    """
    Recursively print AST structure in a readable tree format.

    Args:
        node: A javalang AST node
        indent: Current indentation level
        max_depth: Maximum depth to print (prevents overwhelming output)
    """
    if indent > max_depth:
        print("  " * indent + "... (truncated)")
        return

    prefix = "  " * indent

    if isinstance(node, javalang.ast.Node):
        node_name = type(node).__name__
        attrs = _extract_node_attrs(node)
        attr_str = f" ({', '.join(attrs)})" if attrs else ""
        print(f"{prefix}{node_name}{attr_str}")
        _process_ast_children(node, indent, max_depth)

    elif isinstance(node, str):
        print(f"{prefix}'{node}'")

    elif node is not None:
        print(f"{prefix}{type(node).__name__}: {node}")


def extract_method_body_statements(method_node) -> list:
    """
    Extract individual statements from a method body.
    These could be candidates for "statement-level" chunks.

    Returns list of (statement_type, statement_node) tuples.
    """
    statements = []

    if method_node.body:
        for statement in method_node.body:
            stmt_type = type(statement).__name__
            statements.append((stmt_type, statement))

    return statements


def analyze_method(method_node) -> dict:
    """
    Analyze a method and extract potential chunking information.
    """
    info = {
        'name': method_node.name,
        'return_type': method_node.return_type.name if method_node.return_type else 'void',
        'parameters': [],
        'statements': [],
        'local_variables': [],
        'method_calls': [],
        'control_flow': [],  # if, for, while, etc.
    }

    # Extract parameters
    if method_node.parameters:
        for param in method_node.parameters:
            param_type = param.type.name if hasattr(param.type, 'name') else str(param.type)
            info['parameters'].append(f"{param_type} {param.name}")

    # Extract statements and categorize them
    if method_node.body:
        for stmt in method_node.body:
            stmt_type = type(stmt).__name__
            info['statements'].append(stmt_type)

            # Track local variable declarations
            if isinstance(stmt, javalang.tree.LocalVariableDeclaration):
                for decl in stmt.declarators:
                    info['local_variables'].append(decl.name)

            # Track control flow
            if isinstance(stmt, (javalang.tree.IfStatement,
                                  javalang.tree.ForStatement,
                                  javalang.tree.WhileStatement,
                                  javalang.tree.SwitchStatement,
                                  javalang.tree.TryStatement)):
                info['control_flow'].append(stmt_type)

    # Find all method invocations (recursive search)
    for _, node in method_node.filter(javalang.tree.MethodInvocation):
        info['method_calls'].append(node.member)

    return info


def explore_java_file(java_file_path: str) -> None:
    """
    Main exploration function: parse a Java file and print analysis.
    """
    path = Path(java_file_path)
    print(f"\n{'='*60}")
    print(f"Exploring: {path.name}")
    print(f"{'='*60}\n")

    source = path.read_text(encoding='utf-8')

    # Print raw source for reference
    print("SOURCE CODE:")
    print("-" * 40)
    for i, line in enumerate(source.strip().split('\n'), 1):
        print(f"{i:3}: {line}")
    print("-" * 40)

    # Parse
    try:
        tree = javalang.parse.parse(source)
    except javalang.parser.JavaSyntaxError as e:
        print(f"Parse error: {e}")
        return

    # Find all methods
    methods = list(tree.filter(javalang.tree.MethodDeclaration))

    print(f"\n\nFOUND {len(methods)} METHOD(S):\n")

    for _, method in methods:
        print(f"\n{'─'*40}")
        print(f"METHOD: {method.name}")
        print(f"{'─'*40}")

        # Detailed analysis
        analysis = analyze_method(method)
        print(f"  Return type: {analysis['return_type']}")
        print(f"  Parameters: {', '.join(analysis['parameters']) or 'none'}")
        print(f"  Local vars: {', '.join(analysis['local_variables']) or 'none'}")
        print(f"  Method calls: {', '.join(analysis['method_calls']) or 'none'}")
        print(f"  Control flow: {', '.join(analysis['control_flow']) or 'none'}")
        print(f"  Statement types: {analysis['statements']}")

        # Show AST tree for this method
        print("\n  AST STRUCTURE (max depth=5):")
        print_ast_tree(method, indent=2, max_depth=5)


def extract_statement_code(source_lines: list[str], statement) -> str:
    """
    Extract the actual source code for a statement using its position info.

    javalang AST nodes have 'position' attribute with (line, column).
    This is a simplified extraction - real implementation would need to
    handle multi-line statements and find statement boundaries.
    """
    if not hasattr(statement, 'position') or statement.position is None:
        return "<position unknown>"

    line_num = statement.position.line - 1  # 0-indexed

    if line_num < 0 or line_num >= len(source_lines):
        return "<line out of range>"

    # Simple approach: return the line containing the statement start
    # For multi-line statements, this is incomplete but good for exploration
    return source_lines[line_num].strip()


def demo_chunk_embeddings(java_file_path: str) -> None:
    """
    Show what actual text chunks would look like for embedding.
    """
    path = Path(java_file_path)
    source = path.read_text(encoding='utf-8')
    source_lines = source.split('\n')
    tree = javalang.parse.parse(source)

    print(f"\n{'='*60}")
    print("CHUNK TEXT EXTRACTION DEMO")
    print("(What we would actually feed to the embedding model)")
    print(f"{'='*60}\n")

    for _, method in tree.filter(javalang.tree.MethodDeclaration):
        print(f"\nMethod: {method.name}()")
        print("-" * 40)

        if not method.body:
            print("  (no body)")
            continue

        for i, stmt in enumerate(method.body):
            stmt_type = type(stmt).__name__
            code_text = extract_statement_code(source_lines, stmt)
            print(f"\n  Chunk {i+1} [{stmt_type}]:")
            print(f"    Text: \"{code_text}\"")


def explore_chunking_options(java_file_path: str) -> None:
    """
    Demonstrate different chunking granularities on a Java file.
    """
    path = Path(java_file_path)
    source = path.read_text(encoding='utf-8')
    tree = javalang.parse.parse(source)

    print(f"\n{'='*60}")
    print("CHUNKING OPTIONS DEMO")
    print(f"{'='*60}\n")

    for _, method in tree.filter(javalang.tree.MethodDeclaration):
        print(f"\nMethod: {method.name}()")
        print("-" * 40)

        # Option 1: Whole method as one chunk
        print("\n  OPTION 1 - Whole method = 1 chunk")
        print("    ->> Would embed entire method as single vector")

        # Option 2: Statement-level chunks
        print("\n  OPTION 2 - Statement-level chunks:")
        if method.body:
            for i, stmt in enumerate(method.body):
                stmt_type = type(stmt).__name__
                print(f"    Chunk {i+1}: {stmt_type}")
        else:
            print("    (no body)")

        # Option 3: Signature + body separately
        print("\n  OPTION 3 - Signature vs Body:")
        params = ', '.join(
            f"{p.type.name} {p.name}"
            for p in (method.parameters or [])
        )
        ret_type = method.return_type.name if method.return_type else 'void'
        print(f"    Chunk A (signature): {ret_type} {method.name}({params})")
        print(f"    Chunk B (body): {len(method.body) if method.body else 0} statements")


if __name__ == '__main__':
    import sys

    # Allow passing file path as argument
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
        if Path(file_path).exists():
            explore_java_file(file_path)
            explore_chunking_options(file_path)
            demo_chunk_embeddings(file_path)
        else:
            print(f"File not found: {file_path}")
        sys.exit(0)

    # Default: try sample files from dataset
    sample_files = [
        "../../code-clone-dataset/dataset/base/05/main.java",  # Bank account (more interesting)
        "../../code-clone-dataset/dataset/base/01/main.java",  # Simple calculator
    ]

    for sample in sample_files:
        sample_path = Path(__file__).parent / sample
        if sample_path.exists():
            explore_java_file(str(sample_path))
            explore_chunking_options(str(sample_path))
            demo_chunk_embeddings(str(sample_path))
            break
    else:
        # Fallback: use inline example
        print("Sample files not found. Using inline example.\n")

        EXAMPLE_CODE = '''
public class Example {
    public int calculate(int x, int y) {
        int sum = x + y;
        int product = x * y;
        if (sum > 10) {
            return product;
        }
        return sum;
    }

    public void helper() {
        System.out.println("Hello");
    }
}
'''
        # Write to temp file and analyze
        temp_path = Path("/tmp/example.java")
        temp_path.write_text(EXAMPLE_CODE)
        explore_java_file(str(temp_path))
        explore_chunking_options(str(temp_path))
