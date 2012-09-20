# System library imports.
from chaco.api import PlotGraphicsContext


def save_plot(plot, filename, size=None):
    """ Render a plot to an image on disk.

    The image format is determed from the given filename.
    """
    if size is None:
        size = plot.outer_bounds
    else:
        raise NotImplementedError('Custom sizes not implemented')

    if filename.endswith('.pdf'):
        save_plot_pdf(plot, filename, size)
    elif filename.endswith('.svg'):
        save_plot_svg(plot, filename, size)
    else:
        save_plot_img(plot, filename, size)


def save_plot_img(component, filename, size, dpi=72):
    gc = PlotGraphicsContext(size, dpi=dpi)
    gc.render_component(component)
    gc.save(filename)
        
def save_plot_pdf(component, filename, size, dpi=72):
    from kiva.pdf import GraphicsContext
    from reportlab.pdfgen.canvas import Canvas
    from reportlab.lib.units import inch

    width = (size[0] / float(dpi)) * inch
    height = (size[1] / float(dpi)) * inch
    pagesize = (width, height)
    canvas = Canvas(filename=filename, pagesize=pagesize)

    gc = GraphicsContext(canvas)
    old_backbuffer = component.use_backbuffer
    try:
        component.use_backbuffer = False
        component.draw(gc)
    finally:
        component.use_backbuffer = old_backbuffer
    gc.save()

def save_plot_svg(component, filename, size):
    from chaco.svg_graphics_context import SVGGraphicsContext
    gc = SVGGraphicsContext(size)
    gc.render_component(component)
    gc.save(filename)
