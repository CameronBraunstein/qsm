from utils.file_io.argparser import get_args
from utils.evaluations.evaluator import get_evaluator
from stereo_matchers.stereo_matcher import get_stereo_matcher
from utils.visualizations.matplot_builder import get_matplotlib_builder
from utils.visualizations.qubo_matrix_visualizer import get_qubo_matrix_visualizer

if __name__ == '__main__':
    args = get_args()
    if 'visualize_matrix' in args.actions:
        qmv = get_qubo_matrix_visualizer(args)
        qmv.display_eigenspectrum()
    if 'visualize_paper' in args.actions:
        mplb = get_matplotlib_builder(args)
        mplb.make_custom_png()
    if 'stereo_match' in args.actions:
        sm = get_stereo_matcher(args)
        sm.make_stereo_match()
    if 'evaluate' in args.actions:
        e = get_evaluator(args)
        e.evaluate()
    if 'build_display' in args.actions:
        from utils.visualizations.iviz_builder import get_iviz_builder
        ib = get_iviz_builder(args)
        ib.build_iviz()
