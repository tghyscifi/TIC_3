"""Errors are part of the compiler's public contract, never success placeholders."""
class CompilerError(ValueError):
    def __init__(self, message, *, code='INVALID_MODEL', node=None):
        self.code, self.node = code, node
        super().__init__(f'[{code}]' + (f' {node}:' if node else '') + f' {message}')

class UnsupportedModelError(CompilerError):
    def __init__(self, message, *, node=None):
        super().__init__(message, code='UNSUPPORTED', node=node)

class DependencyError(ImportError):
    pass

