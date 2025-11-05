; function definitions
(function_definition
  name: (identifier) @def.name
  parameters: (parameters) @def.params) @def.node

; call sites with callee names
(call
  function: (identifier) @call.name
  arguments: (argument_list) @call.args) @call.node
