# System library imports.
from chaco.api import PlotGraphicsContext


def save_plot(plot, filename, size=None):
    """ Render a plot to an image on disk.

    The image format is determed from the given filename.
    """
    if size is None:
        size = plot.outer_bounds

    if filename.endswith('.pdf'):
        save_plot_pdf(plot, filename, size)
    elif filename.endswith('.svg'):
        save_plot_svg(plot, filename, size)
    else:
        save_plot_img(plot, filename, size)


def save_plot_img(plot, filename, size, dpi=72):
    gc = PlotGraphicsContext(size, dpi=dpi)
    gc.render_component(plot)
    gc.save(filename)
        
def save_plot_pdf(plot, filename, size):
    from chaco.pdf_graphics_context import PdfPlotGraphicsContext
    gc = PdfPlotGraphicsContext(filename=filename,
                                dest_box=(0.5, 0.5, 5.0, 5.0))
    gc.render_component(plot)
    gc.save()

def save_plot_svg(plot, filename, size):
    from chaco.svg_graphics_context import SVGGraphicsContext
    gc = SVGGraphicsContext(size)
    gc.render_component(plot)
    gc.save(filename)
