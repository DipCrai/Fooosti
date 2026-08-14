import ldm_patched.modules.args_parser as args_parser

args_parser.parser.add_argument("--share", action='store_true', help="Set whether to share on Gradio.")

args_parser.parser.add_argument("--preset", type=str, default=None, help="Apply specified UI preset.")
args_parser.parser.add_argument("--disable-preset-selection", action='store_true',
                                help="Disables preset selection in Gradio.")

args_parser.parser.add_argument("--language", type=str, default='default',
                                help="Translate UI using json files in [language] folder. "
                                  "For example, [--language example] will use [language/example.json] for translation.")

# Keep models in VRAM by default; opt into unloading per task with --offload-from-vram.
args_parser.parser.add_argument("--offload-from-vram", action="store_true",
                                help="Unload models from VRAM after each task instead of keeping them warm. "
                                  "Useful to free VRAM between generations at the cost of a reload per task.")

args_parser.parser.add_argument("--theme", type=str, help="launches the UI with light or dark theme", default=None)
args_parser.parser.add_argument("--disable-image-log", action='store_true',
                                help="Prevent writing images and logs to the outputs folder.")

args_parser.parser.add_argument("--disable-analytics", action='store_true',
                                help="Disables analytics for Gradio.")

args_parser.parser.add_argument("--disable-metadata", action='store_true',
                                help="Disables saving metadata to images.")

args_parser.parser.add_argument("--disable-preset-download", action='store_true',
                                help="Disables downloading models for presets", default=False)

args_parser.parser.add_argument("--disable-enhance-output-sorting", action='store_true',
                                help="Disables enhance output sorting for final image gallery.")

args_parser.parser.add_argument("--enable-auto-describe-image", action='store_true',
                                help="Enables automatic description of uov and enhance image when prompt is empty", default=False)

args_parser.parser.add_argument("--always-download-new-model", action='store_true',
                                help="Always download newer models", default=False)

args_parser.parser.add_argument("--rebuild-hash-cache", help="Generates missing model and LoRA hashes.",
                                type=int, nargs="?", metavar="CPU_NUM_THREADS", const=-1)

args_parser.parser.set_defaults(
    disable_cuda_malloc=True,
    in_browser=True,
    port=None
)

args_parser.args = args_parser.parser.parse_args()

# Keep models warm in VRAM by default. Only unload when explicitly asked to.
args_parser.args.always_offload_from_vram = args_parser.args.offload_from_vram

if args_parser.args.disable_analytics:
    import os
    os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"

if args_parser.args.disable_in_browser:
    args_parser.args.in_browser = False

args = args_parser.args
