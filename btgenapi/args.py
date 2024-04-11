from btgenapi.base_args import add_base_args
import ldm_patched.modules.args_parser as args_parser

add_base_args(args_parser.parser, False)

from args_manager import args_parser

# Override the port default value
args_parser.parser.set_defaults(
    port=9999
)

# Execute args parse again
args_parser.args = args_parser.parser.parse_args()
args = args_parser.args