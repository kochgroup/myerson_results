from matplotlib.lines import Line2D
from matplotlib.patches import Polygon
from matplotlib.pyplot import figure

import seaborn as sns

from rdkit.Chem.Draw.canvasbase import CanvasBase
from IPython.display import SVG, display
from rdkit.Chem import Draw


class NewCanvas(CanvasBase):

  def __init__(self, size, name='', imageType='png'):
    self._name = name
    self.size = size
    dpi = max(size[0], size[1])
    figsize = (int(float(size[0]) / dpi), int(float(size[1]) / dpi))
    self._figure = figure(figsize=figsize)
    self._axes = self._figure.add_axes([0, 0, 1, 1]) # CHANGED HERE
    self._axes.set_xticklabels('')
    self._axes.set_yticklabels('')
    self._dpi = dpi

  def rescalePt(self, p1):
    return [float(p1[0]) / self._dpi, float(self.size[1] - p1[1]) / self._dpi]

  def addCanvasLine(self, p1, p2, color=(0, 0, 0), color2=None, **kwargs):
    canvas = self._axes
    p1 = self.rescalePt(p1)
    p2 = self.rescalePt(p2)
    if color2 and color2 != color:
      mp = (p1[0] + p2[0]) / 2., (p1[1] + p2[1]) / 2.
      canvas.add_line(Line2D((p1[0], mp[0]), (p1[1], mp[1]), color=color, **kwargs))
      canvas.add_line(Line2D((mp[0], p2[0]), (mp[1], p2[1]), color=color2, **kwargs))
    else:
      canvas.add_line(Line2D((p1[0], p2[0]), (p1[1], p2[1]), color=color, **kwargs))

  def addCanvasText(self, text, pos, font, color=(0, 0, 0), **kwargs):
    import re
    pos = self.rescalePt(pos)
    canvas = self._axes
    text = re.sub(r'<.*?>', '', text)
    orientation = kwargs.get('orientation', 'E')
    halign = 'center'
    valign = 'center'
    if orientation == 'E':
      halign = 'left'
    elif orientation == 'W':
      halign = 'right'
    elif orientation == 'S':
      valign = 'top'
    elif orientation == 'N':
      valign = 'bottom'

    annot = canvas.annotate(text, (pos[0], pos[1]), color=color, verticalalignment=valign,
                            horizontalalignment=halign, weight=font.weight, size=font.size * 2.8,
                            family=font.face)

    try:
      bb = annot.get_window_extent(renderer=self._figure.canvas.get_renderer())
      w, h = bb.width, bb.height
      tw, th = canvas.transData.inverted().transform((w, h))
    except AttributeError:
      tw, th = 0.1, 0.1  # <- kludge
    return (tw, th, 0)

  def addCanvasPolygon(self, ps, color=(0, 0, 0), **kwargs):
    canvas = self._axes
    ps = [self.rescalePt(x) for x in ps]
    canvas.add_patch(Polygon(ps, linewidth=0, facecolor=color))

  def addCanvasDashedWedge(self, p1, p2, p3, dash=(2, 2), color=(0, 0, 0), color2=None, **kwargs):
    canvas = self._axes
    dash = (3, 3)
    pts1 = self._getLinePoints(p1, p2, dash)
    pts2 = self._getLinePoints(p1, p3, dash)
    pts1 = [self.rescalePt(p) for p in pts1]
    pts2 = [self.rescalePt(p) for p in pts2]
    if len(pts2) < len(pts1):
      pts2, pts1 = pts1, pts2
    for i in range(len(pts1)):
      if color2 and color2 != color:
        mp = (pts1[i][0] + pts2[i][0]) / 2., (pts1[i][1] + pts2[i][1]) / 2.
        canvas.add_line(Line2D((pts1[i][0], mp[0]), (pts1[i][1], mp[1]), color=color, **kwargs))
        canvas.add_line(Line2D((mp[0], pts2[i][0]), (mp[1], pts2[i][1]), color=color2, **kwargs))
      else:
        canvas.add_line(
          Line2D((pts1[i][0], pts2[i][0]), (pts1[i][1], pts2[i][1]), color=color, **kwargs))


from rdkit import Chem
import numpy
import math
import copy
import functools
def cmp(t1, t2):
    return (t1 < t2) * -1 or (t1 > t2) * 1

periodicTable = Chem.GetPeriodicTable()


class Font(object):

  def __init__(self, face=None, size=None, name=None, weight=None):
    self.face = face or 'sans'
    self.size = size or '12'
    self.weight = weight or 'normal'
    self.name = name


class DrawingOptions(object):
  dotsPerAngstrom = 30
  useFraction = 0.85

  atomLabelFontFace = "sans"
  atomLabelFontSize = 12
  atomLabelMinFontSize = 7
  atomLabelDeuteriumTritium = False

  bondLineWidth = 1.2
  dblBondOffset = .25
  dblBondLengthFrac = .8

  defaultColor = (1, 0, 0)
  selectColor = (1, 0, 0)
  bgColor = (1, 1, 1)

  colorBonds = True
  noCarbonSymbols = True
  includeAtomNumbers = False
  atomNumberOffset = 0
  radicalSymbol = u'\u2219'

  dash = (4, 4)

  wedgeDashedBonds = True
  showUnknownDoubleBonds = True

  # used to adjust overall scaling for molecules that have been laid out with non-standard
  # bond lengths
  coordScale = 1.0

  # elemDict = {
  #   1: (0.55, 0.55, 0.55),
  #   7: (0, 0, 1),
  #   8: (1, 0, 0),
  #   9: (.2, .8, .8),
  #   15: (1, .5, 0),
  #   16: (.8, .8, 0),
  #   17: (0, .8, 0),
  #   35: (.5, .3, .1),
  #   53: (.63, .12, .94),
  #   0: (.5, .5, .5),
  # }
  elemDict = {
    1:  (0, 0, 0),
    7:  (0, 0, 0),
    8:  (0, 0, 0),
    9:  (0, 0, 0),
    15: (0, 0, 0),
    16: (0, 0, 0),
    17: (0, 0, 0),
    35: (0, 0, 0),
    53: (0, 0, 0),
    0:  (0, 0, 0)
  }


class MolDrawing(object):

  def __init__(self, canvas=None, drawingOptions=None):
    self.canvas = canvas
    self.canvasSize = None
    if canvas:
      self.canvasSize = canvas.size
    self.drawingOptions = drawingOptions or DrawingOptions()

    self.atomPs = {}
    self.boundingBoxes = {}

    if self.drawingOptions.bgColor is not None:
      self.canvas.addCanvasPolygon(((0, 0), (canvas.size[0], 0), (canvas.size[0], canvas.size[1]),
                                    (0, canvas.size[1])), color=self.drawingOptions.bgColor,
                                   fill=True, stroke=False)

  def transformPoint(self, pos):
    res = [0, 0]
    res[0] = (pos[0] + self.molTrans[0]
              ) * self.currDotsPerAngstrom * self.drawingOptions.useFraction + self.drawingTrans[0]
    res[1] = self.canvasSize[1] - ((pos[1] + self.molTrans[1]) * self.currDotsPerAngstrom *
                                   self.drawingOptions.useFraction + self.drawingTrans[1])
    return res

  def _getBondOffset(self, p1, p2):
    # get the vector between the points:
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]

    # figure out the angle and the perpendicular:
    ang = math.atan2(dy, dx)
    perp = ang + math.pi / 2.

    # here's the offset for the parallel bond:
    offsetX = math.cos(perp) * self.drawingOptions.dblBondOffset * self.currDotsPerAngstrom
    offsetY = math.sin(perp) * self.drawingOptions.dblBondOffset * self.currDotsPerAngstrom

    return perp, offsetX, offsetY

  def _getOffsetBondPts(self, p1, p2, offsetX, offsetY, lenFrac=None):
    lenFrac = lenFrac or self.drawingOptions.dblBondLengthFrac

    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    # ----
    # now figure out where to start and end it:

    # offset the start point:
    fracP1 = p1[0] + offsetX, p1[1] + offsetY

    # now move a portion of the way along the line to the neighbor:
    frac = (1. - lenFrac) / 2
    fracP1 = fracP1[0] + dx * frac, fracP1[1] + dy * frac
    fracP2 = fracP1[0] + dx * lenFrac, fracP1[1] + dy * lenFrac
    return fracP1, fracP2

  def _offsetDblBond(self, p1, p2, bond, a1, a2, conf, direction=1, lenFrac=None):
    perp, offsetX, offsetY = self._getBondOffset(p1, p2)
    offsetX = offsetX * direction
    offsetY = offsetY * direction

    # if we're a ring bond, we may need to flip over to the other side:
    if bond.IsInRing():
      bondIdx = bond.GetIdx()
      a2Idx = a2.GetIdx()
      # find a ring bond from a1 to an atom other than a2:
      for otherBond in a1.GetBonds():
        if otherBond.GetIdx() != bondIdx and otherBond.IsInRing():
          sharedRing = False
          for ring in self.bondRings:
            if bondIdx in ring and otherBond.GetIdx() in ring:
              sharedRing = True
              break
          if not sharedRing:
            continue
          a3 = otherBond.GetOtherAtom(a1)
          if a3.GetIdx() != a2Idx:
            p3 = self.transformPoint(
              conf.GetAtomPosition(a3.GetIdx()) * self.drawingOptions.coordScale)
            dx2 = p3[0] - p1[0]
            dy2 = p3[1] - p1[1]
            dotP = dx2 * offsetX + dy2 * offsetY
            if dotP < 0:
              perp += math.pi
              offsetX = math.cos(
                perp) * self.drawingOptions.dblBondOffset * self.currDotsPerAngstrom
              offsetY = math.sin(
                perp) * self.drawingOptions.dblBondOffset * self.currDotsPerAngstrom

    fracP1, fracP2 = self._getOffsetBondPts(p1, p2, offsetX, offsetY, lenFrac=lenFrac)
    return fracP1, fracP2

  def _getBondAttachmentCoordinates(self, p1, p2, labelSize):
    newpos = [None, None]
    if labelSize is not None:
      import math
      labelfrac = .373 # extracted from ChemDraw
      labelfrac = .3 
      xdist = abs(p1[0] - p2[0])
      ydist = abs(p1[1] - p2[1])
      xdistremoved = xdist*labelfrac
      ydistremoved = ydist*labelfrac

      x = p1[0] + cmp(p2[0], p1[0])*xdistremoved
      y = p1[1] + cmp(p2[1], p1[1])*ydistremoved

      orient = labelSize[1]
      if orient == 'C':
        newpos = [x,y]
      else:
        newpos = copy.deepcopy(p1)
      # labelSizeOffset = [labelSize[0][0] / 2 + (cmp(p2[0], p1[0]) * labelSize[0][2]),
      #                    labelSize[0][1] / 2]
      # if p1[1] == p2[1]:
      #   newpos[0] = p1[0] + cmp(p2[0], p1[0]) * labelSizeOffset[0]
      # else:
      #   if abs(labelSizeOffset[1] * (p2[0] - p1[0]) / (p2[1] - p1[1])) < labelSizeOffset[0]:
      #     newpos[0] = p1[0] + cmp(p2[0], p1[0]) * abs(labelSizeOffset[1] * (p2[0] - p1[0]) /
      #                                                 (p2[1] - p1[1]))
      #   else:
      #     newpos[0] = p1[0] + cmp(p2[0], p1[0]) * labelSizeOffset[0]
      # if p1[0] == p2[0]:
      #   newpos[1] = p1[1] + cmp(p2[1], p1[1]) * labelSizeOffset[1]
      # else:
      #   if abs(labelSizeOffset[0] * (p1[1] - p2[1]) / (p2[0] - p1[0])) < labelSizeOffset[1]:
      #     newpos[1] = p1[1] + cmp(p2[1], p1[1]) * abs(labelSizeOffset[0] * (p1[1] - p2[1]) /
      #                                                 (p2[0] - p1[0]))
      #   else:
      #     newpos[1] = p1[1] + cmp(p2[1], p1[1]) * labelSizeOffset[1]
    else:
      newpos = copy.deepcopy(p1)
    return newpos

  def _drawWedgedBond(self, bond, pos, nbrPos, width=None, color=None, dash=None):
    width = width or self.drawingOptions.bondLineWidth
    color = color or self.drawingOptions.defaultColor
    _, offsetX, offsetY = self._getBondOffset(pos, nbrPos)
    offsetX *= .75
    offsetY *= .75
    poly = ((pos[0], pos[1]), (nbrPos[0] + offsetX, nbrPos[1] + offsetY),
            (nbrPos[0] - offsetX, nbrPos[1] - offsetY))
    # canvas.drawPolygon(poly,edgeColor=color,edgeWidth=1,fillColor=color,closed=1)
    if not dash:
      self.canvas.addCanvasPolygon(poly, color=color)
    elif self.drawingOptions.wedgeDashedBonds and self.canvas.addCanvasDashedWedge:
      self.canvas.addCanvasDashedWedge(poly[0], poly[1], poly[2], color=color)
    else:
      self.canvas.addCanvasLine(pos, nbrPos, linewidth=width * 2, color=color, dashes=dash)

  def _drawBond(self, bond, atom, nbr, pos, nbrPos, conf, width=None, color=None, color2=None,
                labelSize1=None, labelSize2=None):
    width = width or self.drawingOptions.bondLineWidth
    color = color or self.drawingOptions.defaultColor
    color2 = color2 or self.drawingOptions.defaultColor
    p1_raw = copy.deepcopy(pos)
    p2_raw = copy.deepcopy(nbrPos)
    newpos = self._getBondAttachmentCoordinates(p1_raw, p2_raw, labelSize1)
    newnbrPos = self._getBondAttachmentCoordinates(p2_raw, p1_raw, labelSize2)
    addDefaultLine = functools.partial(self.canvas.addCanvasLine, linewidth=width, color=color,
                                       color2=color2)
    bType = bond.GetBondType()
    if bType == Chem.BondType.SINGLE:
      bDir = bond.GetBondDir()
      if bDir in (Chem.BondDir.BEGINWEDGE, Chem.BondDir.BEGINDASH):
        # if the bond is "backwards", change the drawing direction:
        if bond.GetBeginAtom().GetChiralTag() in (Chem.ChiralType.CHI_TETRAHEDRAL_CW,
                                                  Chem.ChiralType.CHI_TETRAHEDRAL_CCW):
          p1, p2 = newpos, newnbrPos
          wcolor = color
        else:
          p2, p1 = newpos, newnbrPos
          wcolor = color2
        if bDir == Chem.BondDir.BEGINWEDGE:
          self._drawWedgedBond(bond, p1, p2, color=wcolor, width=width)
        elif bDir == Chem.BondDir.BEGINDASH:
          self._drawWedgedBond(bond, p1, p2, color=wcolor, width=width,
                               dash=self.drawingOptions.dash)
      else:
        addDefaultLine(newpos, newnbrPos)
    elif bType == Chem.BondType.DOUBLE:
      crossBond = (self.drawingOptions.showUnknownDoubleBonds and
                   bond.GetStereo() == Chem.BondStereo.STEREOANY)
      if (not crossBond and (bond.IsInRing() or
                             (atom.GetDegree() != 1 and bond.GetOtherAtom(atom).GetDegree() != 1))):
        addDefaultLine(newpos, newnbrPos)
        fp1, fp2 = self._offsetDblBond(newpos, newnbrPos, bond, atom, nbr, conf)
        addDefaultLine(fp1, fp2)
      else:
        fp1, fp2 = self._offsetDblBond(newpos, newnbrPos, bond, atom, nbr, conf, direction=.5,
                                       lenFrac=1.0)
        fp3, fp4 = self._offsetDblBond(newpos, newnbrPos, bond, atom, nbr, conf, direction=-.5,
                                       lenFrac=1.0)
        if crossBond:
          fp2, fp4 = fp4, fp2
        addDefaultLine(fp1, fp2)
        addDefaultLine(fp3, fp4)

    elif bType == Chem.BondType.AROMATIC:
      addDefaultLine(newpos, newnbrPos)
      fp1, fp2 = self._offsetDblBond(newpos, newnbrPos, bond, atom, nbr, conf)
      addDefaultLine(fp1, fp2, dash=self.drawingOptions.dash)
    elif bType == Chem.BondType.TRIPLE:
      addDefaultLine(newpos, newnbrPos)
      fp1, fp2 = self._offsetDblBond(newpos, newnbrPos, bond, atom, nbr, conf)
      addDefaultLine(fp1, fp2)
      fp1, fp2 = self._offsetDblBond(newpos, newnbrPos, bond, atom, nbr, conf, direction=-1)
      addDefaultLine(fp1, fp2)
    else:
      addDefaultLine(newpos, newnbrPos, dash=(1, 2))

  def scaleAndCenter(self, mol, conf, coordCenter=False, canvasSize=None, ignoreHs=False):
    canvasSize = canvasSize or self.canvasSize
    xAccum = 0
    yAccum = 0
    minX = 1e8
    minY = 1e8
    maxX = -1e8
    maxY = -1e8

    nAts = mol.GetNumAtoms()
    for i in range(nAts):
      if ignoreHs and mol.GetAtomWithIdx(i).GetAtomicNum() == 1:
        continue
      pos = conf.GetAtomPosition(i) * self.drawingOptions.coordScale
      xAccum += pos[0]
      yAccum += pos[1]
      minX = min(minX, pos[0])
      minY = min(minY, pos[1])
      maxX = max(maxX, pos[0])
      maxY = max(maxY, pos[1])

    dx = abs(maxX - minX)
    dy = abs(maxY - minY)
    xSize = dx * self.currDotsPerAngstrom
    ySize = dy * self.currDotsPerAngstrom

    if coordCenter:
      molTrans = -xAccum / nAts, -yAccum / nAts
    else:
      molTrans = -(minX + (maxX - minX) / 2), -(minY + (maxY - minY) / 2)
    self.molTrans = molTrans

    if xSize >= .95 * canvasSize[0]:
      scale = .9 * canvasSize[0] / xSize
      xSize *= scale
      ySize *= scale
      self.currDotsPerAngstrom *= scale
      self.currAtomLabelFontSize = max(self.currAtomLabelFontSize * scale,
                                       self.drawingOptions.atomLabelMinFontSize)
    if ySize >= .95 * canvasSize[1]:
      scale = .9 * canvasSize[1] / ySize
      xSize *= scale
      ySize *= scale
      self.currDotsPerAngstrom *= scale
      self.currAtomLabelFontSize = max(self.currAtomLabelFontSize * scale,
                                       self.drawingOptions.atomLabelMinFontSize)
    drawingTrans = canvasSize[0] / 2, canvasSize[1] / 2
    self.drawingTrans = drawingTrans

  def _drawLabel(self, label, pos, baseOffset, font, color=None, **kwargs):
    color = color or self.drawingOptions.defaultColor
    x1 = pos[0]
    y1 = pos[1]
    labelSize = self.canvas.addCanvasText(label, (x1, y1, baseOffset), font, color, **kwargs)
    return labelSize

  def AddMol(self, mol, centerIt=True, molTrans=None, drawingTrans=None, highlightAtoms=[],
             confId=-1, flagCloseContactsDist=2, highlightMap=None, ignoreHs=False,
             highlightBonds=[], **kwargs):
    """Set the molecule to be drawn.

    Parameters:
      hightlightAtoms -- list of atoms to highlight (default [])
      highlightMap -- dictionary of (atom, color) pairs (default None)

    Notes:
      - specifying centerIt will cause molTrans and drawingTrans to be ignored
    """
    conf = mol.GetConformer(confId)
    if 'coordScale' in kwargs:
      self.drawingOptions.coordScale = kwargs['coordScale']

    self.currDotsPerAngstrom = self.drawingOptions.dotsPerAngstrom
    self.currAtomLabelFontSize = self.drawingOptions.atomLabelFontSize
    if centerIt:
      self.scaleAndCenter(mol, conf, ignoreHs=ignoreHs)
    else:
      self.molTrans = molTrans or (0, 0)
      self.drawingTrans = drawingTrans or (0, 0)

    font = Font(face=self.drawingOptions.atomLabelFontFace, size=self.currAtomLabelFontSize)

    obds = None
    if not mol.HasProp('_drawingBondsWedged'):
      # this is going to modify the molecule, get ready to undo that
      obds = [x.GetBondDir() for x in mol.GetBonds()]
      Chem.WedgeMolBonds(mol, conf)

    includeAtomNumbers = kwargs.get('includeAtomNumbers', self.drawingOptions.includeAtomNumbers)
    self.atomPs[mol] = {}
    self.boundingBoxes[mol] = [0] * 4
    self.activeMol = mol
    self.bondRings = mol.GetRingInfo().BondRings()
    labelSizes = {}
    for atom in mol.GetAtoms():
      labelSizes[atom.GetIdx()] = None
      if ignoreHs and atom.GetAtomicNum() == 1:
        drawAtom = False
      else:
        drawAtom = True
      idx = atom.GetIdx()
      pos = self.atomPs[mol].get(idx, None)
      if pos is None:
        pos = self.transformPoint(conf.GetAtomPosition(idx) * self.drawingOptions.coordScale)
        self.atomPs[mol][idx] = pos
        if drawAtom:
          self.boundingBoxes[mol][0] = min(self.boundingBoxes[mol][0], pos[0])
          self.boundingBoxes[mol][1] = min(self.boundingBoxes[mol][1], pos[1])
          self.boundingBoxes[mol][2] = max(self.boundingBoxes[mol][2], pos[0])
          self.boundingBoxes[mol][3] = max(self.boundingBoxes[mol][3], pos[1])

      if not drawAtom:
        continue
      nbrSum = [0, 0]
      for bond in atom.GetBonds():
        nbr = bond.GetOtherAtom(atom)
        if ignoreHs and nbr.GetAtomicNum() == 1:
          continue
        nbrIdx = nbr.GetIdx()
        if nbrIdx > idx:
          nbrPos = self.atomPs[mol].get(nbrIdx, None)
          if nbrPos is None:
            nbrPos = self.transformPoint(
              conf.GetAtomPosition(nbrIdx) * self.drawingOptions.coordScale)
            self.atomPs[mol][nbrIdx] = nbrPos
            self.boundingBoxes[mol][0] = min(self.boundingBoxes[mol][0], nbrPos[0])
            self.boundingBoxes[mol][1] = min(self.boundingBoxes[mol][1], nbrPos[1])
            self.boundingBoxes[mol][2] = max(self.boundingBoxes[mol][2], nbrPos[0])
            self.boundingBoxes[mol][3] = max(self.boundingBoxes[mol][3], nbrPos[1])

        else:
          nbrPos = self.atomPs[mol][nbrIdx]
        nbrSum[0] += nbrPos[0] - pos[0]
        nbrSum[1] += nbrPos[1] - pos[1]

      iso = atom.GetIsotope()
      labelIt = (not self.drawingOptions.noCarbonSymbols or iso or atom.GetAtomicNum() != 6 or
                 atom.GetFormalCharge() != 0 or atom.GetNumRadicalElectrons() or
                 includeAtomNumbers or atom.HasProp('molAtomMapNumber') or atom.GetDegree() == 0)
      orient = ''
      if labelIt:
        baseOffset = 0
        if includeAtomNumbers:
          symbol = str(atom.GetIdx())
          symbolLength = len(symbol)
        else:
          base = atom.GetSymbol()
          if base == 'H' and (iso == 2 or iso == 3) and self.drawingOptions.atomLabelDeuteriumTritium:
            if iso == 2:
              base = 'D'
            else:
              base = 'T'
            iso = 0
          symbolLength = len(base)
          
          nHs = 0
          if not atom.HasQuery():
            nHs = atom.GetTotalNumHs()
          hs = ''
          if nHs == 1:
            hs = 'H'
            symbolLength += 1
          elif nHs > 1:
            hs = 'H<sub>%d</sub>' % nHs
            symbolLength += 1 + len(str(nHs))

          chg = atom.GetFormalCharge()
          if chg == 0:
            chg = ''
          elif chg == 1:
            chg = '+'
          elif chg == -1:
            chg = '-'
          else:
            chg = '%+d' % chg
          symbolLength += len(chg)
          if chg:
            chg = '<sup>%s</sup>' % chg

          if atom.GetNumRadicalElectrons():
            rad = self.drawingOptions.radicalSymbol * atom.GetNumRadicalElectrons()
            rad = '<sup>%s</sup>' % rad
            symbolLength += atom.GetNumRadicalElectrons()
          else:
            rad = ''

          isotope = ''
          isotopeLength = 0
          if iso:
            isotope = '<sup>%d</sup>' % atom.GetIsotope()
            isotopeLength = len(str(atom.GetIsotope()))
            symbolLength += isotopeLength
          mapNum = ''
          mapNumLength = 0
          if atom.HasProp('molAtomMapNumber'):
            mapNum = ':' + atom.GetProp('molAtomMapNumber')
            mapNumLength = 1 + len(str(atom.GetProp('molAtomMapNumber')))
            symbolLength += mapNumLength
          deg = atom.GetDegree()
          # This should be done in a better way in the future:
          # 'baseOffset' should be determined by getting the size of 'isotope' and
          # the size of 'base', or the size of 'mapNum' and the size of 'base'
          # (depending on 'deg' and 'nbrSum[0]') in order to determine the exact
          # position of the base
          
          if deg == 0:
            tSym = periodicTable.GetElementSymbol(atom.GetAtomicNum())
            if tSym in ('O', 'S', 'Se', 'Te', 'F', 'Cl', 'Br', 'I', 'At'):
              symbol = '%s%s%s%s%s%s' % (hs, isotope, base, chg, rad, mapNum)
            else:
              symbol = '%s%s%s%s%s%s' % (isotope, base, hs, chg, rad, mapNum)
          elif deg > 1 or nbrSum[0] < 1:
            symbol = '%s%s%s%s%s%s' % (isotope, base, hs, chg, rad, mapNum)
            baseOffset = 0.5 - (isotopeLength + len(base) / 2.) / symbolLength
          else:
            symbol = '%s%s%s%s%s%s' % (rad, chg, hs, isotope, base, mapNum)
            baseOffset = -0.5 + (mapNumLength + len(base) / 2.) / symbolLength
            
          if deg == 1:
            if abs(nbrSum[1]) > 1:
              islope = nbrSum[0] / abs(nbrSum[1])
            else:
              islope = nbrSum[0]
            if abs(islope) > .3:
              if islope > 0:
                orient = 'W'
              else:
                orient = 'E'
            elif abs(nbrSum[1]) > 10:
              if nbrSum[1] > 0:
                orient = 'N'
              else:
                orient = 'S'
          else:
            orient = 'C'
        
        if highlightMap and idx in highlightMap:
          color = highlightMap[idx]
        elif highlightAtoms and idx in highlightAtoms:
          color = self.drawingOptions.selectColor
        else:
          color = self.drawingOptions.elemDict.get(atom.GetAtomicNum(), (0, 0, 0))
        labelSize = self._drawLabel(symbol, pos, baseOffset, font, color=color, orientation=orient)
        labelSizes[atom.GetIdx()] = [labelSize, orient]

    for bond in mol.GetBonds():
      atom, idx = bond.GetBeginAtom(), bond.GetBeginAtomIdx()
      nbr, nbrIdx = bond.GetEndAtom(), bond.GetEndAtomIdx()
      pos = self.atomPs[mol].get(idx, None)
      nbrPos = self.atomPs[mol].get(nbrIdx, None)
      if highlightBonds and bond.GetIdx() in highlightBonds:
        width = 2.0 * self.drawingOptions.bondLineWidth
        color = self.drawingOptions.selectColor
        color2 = self.drawingOptions.selectColor
      elif highlightAtoms and idx in highlightAtoms and nbrIdx in highlightAtoms:
        width = 2.0 * self.drawingOptions.bondLineWidth
        color = self.drawingOptions.selectColor
        color2 = self.drawingOptions.selectColor
      elif highlightMap is not None and idx in highlightMap and nbrIdx in highlightMap:
        width = 2.0 * self.drawingOptions.bondLineWidth
        color = highlightMap[idx]
        color2 = highlightMap[nbrIdx]
      else:
        width = self.drawingOptions.bondLineWidth
        if self.drawingOptions.colorBonds:
          color = self.drawingOptions.elemDict.get(atom.GetAtomicNum(), (0, 0, 0))
          color2 = self.drawingOptions.elemDict.get(nbr.GetAtomicNum(), (0, 0, 0))
        else:
          color = self.drawingOptions.defaultColor
          color2 = color
      self._drawBond(bond, atom, nbr, pos, nbrPos, conf, color=color, width=width, color2=color2,
                     labelSize1=labelSizes[idx], labelSize2=labelSizes[nbrIdx])

  # if we modified the bond wedging state, undo those changes now
    if obds:
      for i, d in enumerate(obds):
        mol.GetBondWithIdx(i).SetBondDir(d)

    if flagCloseContactsDist > 0:
      tol = flagCloseContactsDist * flagCloseContactsDist
      for i, _ in enumerate(mol.GetAtoms()):
        pi = numpy.array(self.atomPs[mol][i])
        for j in range(i + 1, mol.GetNumAtoms()):
          pj = numpy.array(self.atomPs[mol][j])
          d = pj - pi
          dist2 = d[0] * d[0] + d[1] * d[1]
          if dist2 <= tol:
            self.canvas.addCanvasPolygon(
              ((pi[0] - 2 * flagCloseContactsDist, pi[1] - 2 * flagCloseContactsDist),
               (pi[0] + 2 * flagCloseContactsDist, pi[1] - 2 * flagCloseContactsDist),
               (pi[0] + 2 * flagCloseContactsDist, pi[1] + 2 * flagCloseContactsDist),
               (pi[0] - 2 * flagCloseContactsDist, pi[1] + 2 * flagCloseContactsDist)),
              color=(1., 0, 0), fill=False, stroke=True)



def NewMolToMPL(mol, size=(300, 300), kekulize=True, wedgeBonds=True, imageType=None, fitImage=False,
             options=None, **kwargs):
  """ Generates a drawing of a molecule on a matplotlib canvas
  """
  if not mol:
    raise ValueError('Null molecule provided')
  # from rdkit.Chem.Draw.mplCanvas import Canvas
  canvas = NewCanvas(size)
  if options is None:
    options = DrawingOptions()
    options.bgColor = None
  if fitImage:
    options.dotsPerAngstrom = int(min(size) / 10)
  options.wedgeDashedBonds = wedgeBonds
  drawer = MolDrawing(canvas=canvas, drawingOptions=options)
  omol = mol
  if kekulize:
    from rdkit import Chem
    mol = Chem.Mol(mol.ToBinary())
    Chem.Kekulize(mol)

  if not mol.GetNumConformers():
    from rdkit.Chem import AllChem
    AllChem.Compute2DCoords(mol)

  drawer.AddMol(mol, **kwargs)
  omol._atomPs = drawer.atomPs[mol]
  for k, v in omol._atomPs.items():
    omol._atomPs[k] = canvas.rescalePt(v)
  canvas._figure.set_size_inches(float(size[0]) / 100, float(size[1]) / 100)
  return canvas._figure


try:
  from matplotlib import cm
  from matplotlib.colors import LinearSegmentedColormap
except ImportError:
  cm = None
except RuntimeError:
  cm = None

import numpy

from rdkit import Chem
from rdkit import Geometry
from rdkit.Chem import Draw
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem import rdDepictor
import math

def NewSimilarityMapFromWeights(mol, weights, colorMap=None, scale=-1, size=(250, 250), sigma=None,
                                coordScale=1.5, step=0.01, colors='k', contourLines=10, alpha=0.5,
                                draw2d=None, **kwargs):
  """
    Generates the similarity map for a molecule given the atomic weights.
    Parameters:
      mol -- the molecule of interest
      colorMap -- the matplotlib color map scheme, default is custom PiWG color map
      scale -- the scaling: scale < 0 -> the absolute maximum weight is used as maximum scale
                            scale = double -> this is the maximum scale
      size -- the size of the figure
      sigma -- the sigma for the Gaussians
      coordScale -- scaling factor for the coordinates
      step -- the step for calcAtomGaussian
      colors -- color of the contour lines
      contourLines -- if integer number N: N contour lines are drawn
                      if list(numbers): contour lines at these numbers are drawn
      alpha -- the alpha blending value for the contour lines
      kwargs -- additional arguments for drawing
    """
  if mol.GetNumAtoms() < 2:
    raise ValueError("too few atoms")

  if draw2d is not None:
    mol = rdMolDraw2D.PrepareMolForDrawing(mol, addChiralHs=False)
    if not mol.GetNumConformers():
      rdDepictor.Compute2DCoords(mol)
    if sigma is None:
      if mol.GetNumBonds() > 0:
        bond = mol.GetBondWithIdx(0)
        idx1 = bond.GetBeginAtomIdx()
        idx2 = bond.GetEndAtomIdx()
        sigma = 0.3 * (mol.GetConformer().GetAtomPosition(idx1) -
                       mol.GetConformer().GetAtomPosition(idx2)).Length()
      else:
        sigma = 0.3 * (mol.GetConformer().GetAtomPosition(0) -
                       mol.GetConformer().GetAtomPosition(1)).Length()
      sigma = round(sigma, 2)
    
    sigmas = [sigma] * mol.GetNumAtoms()
    locs = []
    for i in range(mol.GetNumAtoms()):
      p = mol.GetConformer().GetAtomPosition(i)
      locs.append(Geometry.Point2D(p.x, p.y))
    draw2d.ClearDrawing()
    ps = Draw.ContourParams()
    ps.fillGrid = True
    ps.gridResolution = 0.1
    ps.extraGridPadding = 0.5
    
    if colorMap is not None:
      if cm is not None and isinstance(colorMap, type(cm.Blues)):
        # it's a matplotlib colormap:
        clrs = [tuple(x) for x in colorMap([0, 0.5, 1])]
      elif type(colorMap)==str:
        if cm is None:
          raise ValueError("cannot provide named colormaps unless matplotlib is installed")
        clrs = [tuple(x) for x in cm.get_cmap(colorMap)([0, 0.5, 1])]
      else:
        clrs = [colorMap[0], colorMap[1], colorMap[2]]
      ps.setColourMap(clrs)

    Draw.ContourAndDrawGaussians(draw2d, locs, weights, sigmas, nContours=contourLines, params=ps)
    draw2d.drawOptions().clearBackground = False
    draw2d.DrawMolecule(mol)
    return draw2d

  fig = NewMolToMPL(mol, coordScale=coordScale, size=size, **kwargs)
  if sigma is None:
    if mol.GetNumBonds() > 0:
      bond = mol.GetBondWithIdx(0)
      idx1 = bond.GetBeginAtomIdx()
      idx2 = bond.GetEndAtomIdx()
      sigma = 0.3 * math.sqrt(
        sum([(mol._atomPs[idx1][i] - mol._atomPs[idx2][i])**2 for i in range(2)]))
    else:
      sigma = 0.3 * \
          math.sqrt(sum([(mol._atomPs[0][i] - mol._atomPs[1][i])**2 for i in range(2)]))
    sigma = round(sigma, 2)
  x, y, z = Draw.calcAtomGaussians(mol, sigma, weights=weights, step=step)
  
  # scaling
  if scale <= 0.0:
    maxScale = max(math.fabs(numpy.min(z)), math.fabs(numpy.max(z)))
  else:
    maxScale = scale
  
  # coloring
  if colorMap is None:
    if cm is None:
      raise RuntimeError("matplotlib failed to import")
    PiYG_cmap = cm.get_cmap('PiYG', 2)
    colorMap = LinearSegmentedColormap.from_list(
      'PiWG', [PiYG_cmap(0), (1.0, 1.0, 1.0), PiYG_cmap(1)], N=255)

  fig.axes[0].imshow(z, cmap=colorMap, interpolation='bilinear', origin='lower',
                     extent=(0, 1, 0, 1), vmin=-maxScale, vmax=maxScale)
  # contour lines
  # only draw them when at least one weight is not zero
  if len([w for w in weights if w != 0.0]):
    contourset = fig.axes[0].contour(x, y, z, contourLines, colors=colors, alpha=alpha, **kwargs)
    for j, c in enumerate(contourset.collections):
      if contourset.levels[j] == 0.0:
        c.set_linewidth(0.0)
      elif contourset.levels[j] < 0:
        c.set_dashes([(0, (3.0, 3.0))])
  fig.axes[0].set_axis_off()
  
  return fig


# custom colormaps
from matplotlib.colors import LinearSegmentedColormap
darkblue_pink = LinearSegmentedColormap.from_list('darkblue_pink', (
    # Edit this gradient at https://eltos.github.io/gradient/#Random%20gradient%206784=00578A-FFFFFF-AF0078
    (0.000, (0.000, 0.341, 0.541)),
    (0.500, (1.000, 1.000, 1.000)),
    (1.000, (0.686, 0.000, 0.471))))

darkblue_green = LinearSegmentedColormap.from_list('darkblue_green', (
    # Edit this gradient at https://eltos.github.io/gradient/#Random%20gradient%206784=00578A-FFFFFF-7AB51D
    (0.000, (0.000, 0.341, 0.541)),
    (0.500, (1.000, 1.000, 1.000)),
    (1.000, (0.478, 0.710, 0.114))))
lightblue_green = LinearSegmentedColormap.from_list('lightblue_green', (
    # Edit this gradient at https://eltos.github.io/gradient/#Random%20gradient%206784=009DD1-FFFFFF-7AB51D
    (0.000, (0.000, 0.616, 0.820)),
    (0.500, (1.000, 1.000, 1.000)),
    (1.000, (0.478, 0.710, 0.114))))

lightblue_redish = LinearSegmentedColormap.from_list('lightblue_redish ', (
    # Edit this gradient at https://eltos.github.io/gradient/#Random%20gradient%206784=009DD1-FFFFFF-A1051D
    (0.000, (0.000, 0.616, 0.820)),
    (0.500, (1.000, 1.000, 1.000)),
    (1.000, (0.631, 0.020, 0.114))))

# use this colormap
lightblue_w_orange = LinearSegmentedColormap.from_list('lightblue_w_orange', (
    # Edit this gradient at https://eltos.github.io/gradient/#Random%20gradient%206784=0:009DD1-47.5:FFFFFF-50:FFFFFF-52.5:FFFFFF-100:FF6400
    (0.000, (0.000, 0.616, 0.820)),
    (0.475, (1.000, 1.000, 1.000)),
    (0.500, (1.000, 1.000, 1.000)),
    (0.525, (1.000, 1.000, 1.000)),
    (1.000, (1.000, 0.392, 0.000))))

lightblue_orange = LinearSegmentedColormap.from_list('lightblue_orange', (
    # Edit this gradient at https://eltos.github.io/gradient/#Random%20gradient%206784=009DD1-FFFFFF-FF6400
    (0.000, (0.000, 0.616, 0.820)),
    (0.500, (1.000, 1.000, 1.000)),
    (1.000, (1.000, 0.392, 0.000))))

blue_w_red = LinearSegmentedColormap.from_list('blue_w_red', (
    # Edit this gradient at https://eltos.github.io/gradient/#Random%20gradient%206784=0:003FFF-47.5:FFFFFF-50:FFFFFF-52.5:FFFFFF-100:FF0005
    (0.000, (0.000, 0.247, 1.000)),
    (0.475, (1.000, 1.000, 1.000)),
    (0.500, (1.000, 1.000, 1.000)),
    (0.525, (1.000, 1.000, 1.000)),
    (1.000, (1.000, 0.000, 0.020))))
blue_red = LinearSegmentedColormap.from_list('blue_w_red', (
    # Edit this gradient at https://eltos.github.io/gradient/#Random%20gradient%206784=0:003FFF-47.5:FFFFFF-50:FFFFFF-52.5:FFFFFF-100:FF0005
    (0.000, (0.000, 0.247, 1.000)),
    (0.500, (1.000, 1.000, 1.000)),
    (1.000, (1.000, 0.000, 0.020))))
yellow_blue = LinearSegmentedColormap.from_list('Random gradient 6784', (
    # Edit this gradient at https://eltos.github.io/gradient/#Random%20gradient%206784=D8E700-FFFFFF-000DDC
    (0.000, (0.847, 0.906, 0.000)),
    (0.500, (1.000, 1.000, 1.000)),
    (1.000, (0.000, 0.051, 0.863))))
green_purple = LinearSegmentedColormap.from_list('Random gradient 6784', (
    # Edit this gradient at https://eltos.github.io/gradient/#Random%20gradient%206784=238A00-FFFFFF-5500AF
    (0.000, (0.137, 0.541, 0.000)),
    (0.500, (1.000, 1.000, 1.000)),
    (1.000, (0.333, 0.000, 0.686))))

import io
from PIL import Image
from matplotlib import pyplot as plt
import matplotlib as mpl

def draw_with_weights(smiles: str, weights: list, save_instead=False, cmap=lightblue_orange,
    size=(200, 200), save_format='png'):
    mol = Chem.MolFromSmiles(smiles)
    # cmap = mpl.colormaps[cmap]
    fig = NewSimilarityMapFromWeights(mol, weights, contourLines=0, size=size,
        colorMap=cmap)

    cm_unit = 1/2.54
    fig.set_size_inches(7.5*cm_unit, 7.5*cm_unit) # rescale figure
    if save_instead:
        if isinstance(save_instead, str):
            fig.savefig(save_instead+'.'+save_format, format=save_format, facecolor='white')
        else:
            fig.savefig('TESTFIGURE.pdf', format='pdf', facecolor='white')
    else:
        buf = io.BytesIO()
        fig.savefig(buf, format='png', facecolor='white')
        im = Image.open(buf)
        display(im) # type: ignore
        buf.close()
    plt.close()

def draw_colorbar(weights, save_instead=False, cmap=lightblue_orange, save_format='png'):
    cm_unit = 1/2.54
    fig, ax = plt.subplots(figsize=(7.5*cm_unit,1*cm_unit))
    wmin = min(weights)
    wmax = max(weights)
    vmin = min(wmin, -wmax)
    vmax = max(wmax, -wmin)
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    # norm = mpl.colors.Normalize(vmin=min(weights), vmax=max(weights)) # wrong normalization
    fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), cax=ax, orientation='horizontal')
    fig.subplots_adjust(bottom=0.5)
    fig.set_size_inches(7.5*cm_unit, 1.5*cm_unit) # rescale figure
    if save_instead:
        if isinstance(save_instead, str):
            fig.savefig(save_instead+'.'+save_format, format=save_format, facecolor='white')
        else:
            fig.savefig('TESTFIGURE.pdf', format='pdf', facecolor='white')
    else:
        buf = io.BytesIO()
        fig.savefig(buf, format='png', facecolor='white')
        im = Image.open(buf)
        display(im) # type: ignore
        buf.close()
    plt.close()

import numpy as np

def find_example_explanations(results, num_best=10, num_worst=10, num_random=10,
    print_score=False):
    '''Find graphs with the best, worst and random predictions. Returns arrays 
    of indices/keys.'''
    labels = np.array([results[x]['label'] for x in results])
    predictions = np.array([results[x]['pred'] for x in results])
    compare = np.abs(labels - predictions)
    sorted_indices = np.argsort(compare) # sort the array for the 

    best = [x for x in sorted_indices[:num_best]]
    worst = [x for x in sorted_indices[-num_worst:]]
    random = [x for x in np.random.choice(sorted_indices[num_best:-num_worst], 
        size=num_random, replace=False)]
    if print_score:
        print(f'best\n{[compare[x] for x in best]}')
        print(f'worst\n{[compare[x] for x in worst]}')
        print(f'random\n{[compare[x] for x in random]}')
    return best, worst, random

def draw_density(arr_1, arr_2, title, fig_w=7.5, fig_h=5):
    cm_unit = 1/2.54
    fig, ax = plt.subplots(figsize=(fig_w*cm_unit, fig_h*cm_unit), dpi=200)
    sns.kdeplot(arr_1, cut=0, ax=ax, color='#099dd1', legend=True)
    sns.kdeplot(arr_2, cut=0, ax=ax, color='#333f48')
    custom_lines = [Line2D([0], [0], color='#099dd1'),
                Line2D([0], [0], color='#333f48')]
    
    ax.legend(custom_lines, ['GCNConv', 'GATConv'])
    ax.set_xlabel('Pearson\'s r')
    ax.set_ylabel('Density')
    ax.tick_params(color='grey')
    ax.set_xlim((-1, 1))


    pad_left = 2.5*cm_unit #+ .4*cm_unit 
    pad_bottom = 2.8*cm_unit #+ .2*cm_unit 
    pad_right = .8*cm_unit
    pad_top = .8*cm_unit
    # w_space = 0.25*cm_unit
    # h_space = 0.5*cm_unit

    plt.subplots_adjust(0 + pad_left/fig_w, 
                        0 + pad_bottom/fig_h, 
                        1 - pad_right/fig_w, 
                        1 - pad_top/fig_h)#,
                        # 0 + 2*w_space/fig_w, 
                        # 0 + 2*h_space/fig_h)
    if not title:
        buf = io.BytesIO()
        plt.savefig(buf, format='png', facecolor='white')
        im = Image.open(buf)
        display(im)
        buf.close()
    else:
        plt.savefig(fname=title+'.pdf')
        print("figure saved")
    plt.close()


def draw_bw_with_indices(smiles, title=False, size=(200, 200)):
    mol = Chem.MolFromSmiles(smiles)
    d = Draw.rdMolDraw2D.MolDraw2DSVG(size[0], size[1]) 
    d.drawOptions().useBWAtomPalette()
    d.drawOptions().addAtomIndices = True
    d.DrawMolecule(mol)
    d.FinishDrawing()
    image = d.GetDrawingText() 
    if title:
        with open(title+'.svg', 'w') as f:
            f.write(image)
    else:
        SVG(image)

def draw_bw_with_values(smiles, values:list, title=False, size=(200, 200)):
    mol = Chem.MolFromSmiles(smiles)
    for i, a in enumerate(mol.GetAtoms()):
      a.SetProp('atomNote', f"{values[i]:.2f}")
    d = Draw.rdMolDraw2D.MolDraw2DSVG(size[0], size[1]) 
    d.drawOptions().useBWAtomPalette()
    d.DrawMolecule(mol)
    d.FinishDrawing()
    image = d.GetDrawingText() 
    if title:
        with open(title+'.svg', 'w') as f:
            f.write(image)
    else:
        SVG(image)