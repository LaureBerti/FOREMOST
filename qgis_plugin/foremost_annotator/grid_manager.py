"""
grid_manager.py — Create and manage the N×N planning-unit vector layer.

The grid is a QGIS memory vector layer (Polygon) with one feature per cell.
Attributes: row (int), col (int), class_code (int), cost (double).
"""

from qgis.core import (
    QgsVectorLayer, QgsFeature, QgsGeometry, QgsRectangle,
    QgsField, QgsCategorizedSymbolRenderer, QgsRendererCategory,
    QgsFillSymbol, QgsProject, QgsWkbTypes, QgsCoordinateReferenceSystem,
    QgsGraduatedSymbolRenderer, QgsRendererRange,
    QgsColorRampShader, QgsStyle,
)
from qgis.PyQt.QtCore import QVariant
from qgis.PyQt.QtGui import QColor

from .constants import (
    CLASS_NONE, CLASS_HAB, CLASS_RA, CLASS_NR,
    CLASS_LABEL, CLASS_FILL, CLASS_STROKE,
    FLD_ROW, FLD_COL, FLD_CLASS, FLD_COST,
)

LAYER_NAME      = "FOREMOST Grid"
COST_LAYER_NAME = "FOREMOST Cost"


class GridManager:
    """Create, update, and query the N×N planning-unit layer."""

    FIELDS = [
        QgsField("row",        QVariant.Int),
        QgsField("col",        QVariant.Int),
        QgsField("class_code", QVariant.Int),
        QgsField("cost",       QVariant.Double),
    ]

    def __init__(self):
        self.layer:      QgsVectorLayer | None = None
        self.cost_layer: QgsVectorLayer | None = None
        self.N:      int   = 0
        self.extent: QgsRectangle | None = None
        self.cell_w: float = 0.0
        self.cell_h: float = 0.0

    # ── creation ──────────────────────────────────────────────────────────────

    def create(self, extent: QgsRectangle, N: int, crs: QgsCoordinateReferenceSystem) -> QgsVectorLayer:
        """Create (or replace) the N×N grid layer over *extent*."""
        # Remove old layer if present
        self.remove()

        self.N = N

        # Force a square extent so cells are physically square on the map.
        cx   = (extent.xMinimum() + extent.xMaximum()) / 2
        cy   = (extent.yMinimum() + extent.yMaximum()) / 2
        half = max(extent.width(), extent.height()) / 2
        sq   = QgsRectangle(cx - half, cy - half, cx + half, cy + half)

        self.extent = sq
        self.cell_w = sq.width()  / N   # == cell_h (square)
        self.cell_h = sq.height() / N

        layer = QgsVectorLayer("Polygon", LAYER_NAME, "memory")
        layer.setCrs(crs)

        pr = layer.dataProvider()
        pr.addAttributes(self.FIELDS)
        layer.updateFields()

        features = []
        for row in range(N):
            for col in range(N):
                x0 = sq.xMinimum() + col       * self.cell_w
                x1 = sq.xMinimum() + (col + 1) * self.cell_w
                y1 = sq.yMaximum() - row       * self.cell_h
                y0 = sq.yMaximum() - (row + 1) * self.cell_h
                feat = QgsFeature()
                feat.setGeometry(QgsGeometry.fromRect(QgsRectangle(x0, y0, x1, y1)))
                feat.setAttributes([row, col, CLASS_NONE, 0.0])
                features.append(feat)

        pr.addFeatures(features)
        layer.updateExtents()
        _apply_renderer(layer)
        layer.setOpacity(0.75)

        QgsProject.instance().addMapLayer(layer)
        self.layer = layer
        return layer

    # ── queries ───────────────────────────────────────────────────────────────

    def feature_at(self, map_point) -> QgsFeature | None:
        """Return the grid cell containing *map_point*, or None."""
        if self.layer is None:
            return None
        tiny = QgsRectangle(map_point.x() - 1e-6, map_point.y() - 1e-6,
                            map_point.x() + 1e-6, map_point.y() + 1e-6)
        req = self.layer.getFeatures(tiny)
        return next(req, None)

    def row_col(self, map_point) -> tuple[int, int] | None:
        """Return (row, col) for *map_point*, or None if outside grid."""
        if self.extent is None:
            return None
        col = int((map_point.x() - self.extent.xMinimum()) / self.cell_w)
        row = int((self.extent.yMaximum() - map_point.y()) / self.cell_h)
        if 0 <= row < self.N and 0 <= col < self.N:
            return row, col
        return None

    def set_cell(self, feature: QgsFeature, class_code: int, cost: float = 0.0):
        """Update class and cost of one cell (editing must be active)."""
        fid = feature.id()
        self.layer.changeAttributeValue(fid, FLD_CLASS, class_code)
        self.layer.changeAttributeValue(fid, FLD_COST,  cost if class_code == CLASS_RA else 0.0)
        self.layer.triggerRepaint()

    def set_class_bulk(self, class_code: int, cost: float = 0.0):
        """Set all cells to *class_code* (used for global reset)."""
        if self.layer is None:
            return
        # Build the full attribute map in one pass, then push as a single
        # provider call — avoids N*2 individual signal-emitting change calls.
        attrs = {
            feat.id(): {FLD_CLASS: class_code, FLD_COST: 0.0}
            for feat in self.layer.getFeatures()
        }
        self.layer.dataProvider().changeAttributeValues(attrs)
        self.layer.reload()
        self.layer.triggerRepaint()

    # ── session ───────────────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        """Serialise grid state (labels + costs) to a JSON-safe dict."""
        if self.layer is None:
            return {}
        labels = {}
        costs  = {}
        for feat in self.layer.getFeatures():
            key = f"{feat['row']},{feat['col']}"
            labels[key] = int(feat["class_code"])
            costs[key]  = float(feat["cost"])
        return {
            "N":      self.N,
            "labels": labels,
            "costs":  costs,
            "georef": {
                "extent": [self.extent.xMinimum(), self.extent.yMinimum(),
                           self.extent.xMaximum(), self.extent.yMaximum()],
                "crs":    self.layer.crs().authid(),
            },
        }

    def from_dict(self, data: dict):
        """Restore labels/costs from a previously saved dict."""
        if self.layer is None:
            return
        labels = data.get("labels", {})
        costs  = data.get("costs",  {})
        attrs = {}
        for feat in self.layer.getFeatures():
            key  = f"{feat['row']},{feat['col']}"
            attrs[feat.id()] = {
                FLD_CLASS: int(labels.get(key, CLASS_NONE)),
                FLD_COST:  float(costs.get(key, 0.0)),
            }
        if not attrs:
            return
        if self.layer.isEditable():
            self.layer.rollBack()
        # Update provider directly (bypasses edit buffer → fast batch write).
        # Then rebuild the renderer from scratch so the symbol cache is fully
        # invalidated — commitChanges() on an empty buffer does not reliably
        # trigger cache invalidation, but setRenderer() always does.
        self.layer.dataProvider().changeAttributeValues(attrs)
        self.layer.reload()
        _apply_renderer(self.layer)
        self.layer.triggerRepaint()

    # ── cost visualisation layer ──────────────────────────────────────────────

    def refresh_cost_layer(self) -> QgsVectorLayer | None:
        """
        Create (or replace) a graduated-colour overlay showing restoration
        cost per Restorable cell.  Cells with cost = 0 are skipped.
        Returns the layer, or None if there is nothing to show.
        """
        if self.layer is None:
            return None

        # collect (geometry, cost) for non-zero RA cells
        rows = []
        for feat in self.layer.getFeatures():
            cost = float(feat[FLD_COST])
            if cost > 0:
                rows.append((QgsGeometry(feat.geometry()), cost))

        # remove old cost layer
        if self.cost_layer is not None:
            try:
                QgsProject.instance().removeMapLayer(self.cost_layer.id())
            except Exception:
                pass
            self.cost_layer = None

        if not rows:
            return None

        # build memory layer
        clayer = QgsVectorLayer("Polygon", COST_LAYER_NAME, "memory")
        clayer.setCrs(self.layer.crs())
        pr = clayer.dataProvider()
        pr.addAttributes([QgsField("cost", QVariant.Double)])
        clayer.updateFields()

        feats = []
        for geom, cost in rows:
            f = QgsFeature()
            f.setGeometry(geom)
            f.setAttributes([cost])
            feats.append(f)
        pr.addFeatures(feats)
        clayer.updateExtents()

        # graduated renderer: 5 equal-interval classes, YlOrRd palette
        costs   = [c for _, c in rows]
        lo, hi  = min(costs), max(costs)
        n_cls   = 5
        step    = (hi - lo) / n_cls if hi > lo else 1.0
        ramp    = [
            QColor("#ffffb2"), QColor("#fecc5c"),
            QColor("#fd8d3c"), QColor("#f03b20"), QColor("#bd0026"),
        ]
        ranges = []
        for i in range(n_cls):
            r_lo  = lo + i * step
            r_hi  = lo + (i + 1) * step
            sym   = QgsFillSymbol.createSimple({
                "color":         ramp[i].name(),
                "outline_color": "#888888",
                "outline_width": "0.1",
            })
            ranges.append(QgsRendererRange(r_lo, r_hi, sym, f"{r_lo:,.0f} – {r_hi:,.0f}"))
        renderer = QgsGraduatedSymbolRenderer("cost", ranges)
        clayer.setRenderer(renderer)
        clayer.setOpacity(0.85)

        # Add and then move immediately above the grid layer in the tree
        QgsProject.instance().addMapLayer(clayer, False)
        root = QgsProject.instance().layerTreeRoot()
        grid_node = root.findLayer(self.layer.id())
        if grid_node:
            parent = grid_node.parent()
            idx    = parent.children().index(grid_node)
            parent.insertLayer(idx, clayer)
        else:
            root.insertLayer(0, clayer)

        self.cost_layer = clayer
        return clayer

    # ── cleanup ───────────────────────────────────────────────────────────────

    def remove(self):
        """Remove ALL FOREMOST grid/cost layers from the project."""
        # Remove tracked references first
        for attr in ("cost_layer", "layer"):
            lyr = getattr(self, attr)
            if lyr is not None:
                try:
                    QgsProject.instance().removeMapLayer(lyr.id())
                except Exception:
                    pass
                setattr(self, attr, None)

        # Remove any stale layers that share the same name (handles plugin
        # reload, zoom-triggered duplicates, and manual layer duplication).
        for name in (COST_LAYER_NAME, LAYER_NAME):
            for lyr in list(QgsProject.instance().mapLayersByName(name)):
                try:
                    QgsProject.instance().removeMapLayer(lyr.id())
                except Exception:
                    pass

    @property
    def active(self) -> bool:
        if self.layer is None:
            return False
        try:
            valid = self.layer.isValid()
        except RuntimeError:
            # C++ QgsVectorLayer object was deleted (e.g. user removed the layer manually)
            self.layer = None
            return False
        if not valid:
            self.layer = None
        return valid

    def cell_size_m(self, target_crs_auth: str = "EPSG:3857") -> float:
        """Approximate cell width in metres (projects extent to metres-based CRS)."""
        if self.layer is None:
            return 100.0
        try:
            from qgis.core import QgsCoordinateTransform
            src_crs = self.layer.crs()
            dst_crs = QgsCoordinateReferenceSystem(target_crs_auth)
            xform   = QgsCoordinateTransform(src_crs, dst_crs, QgsProject.instance())
            p0 = xform.transform(self.extent.xMinimum(), self.extent.yMinimum())
            p1 = xform.transform(self.extent.xMaximum(), self.extent.yMinimum())
            total_m = ((p1.x() - p0.x()) ** 2 + (p1.y() - p0.y()) ** 2) ** 0.5
            return total_m / self.N
        except Exception:
            return self.cell_w


# ── renderer ──────────────────────────────────────────────────────────────────

def _apply_renderer(layer: QgsVectorLayer):
    cats = []
    for cls in (CLASS_NONE, CLASS_HAB, CLASS_RA, CLASS_NR):
        sym = QgsFillSymbol.createSimple({
            "color":         CLASS_FILL[cls],
            "outline_color": CLASS_STROKE[cls],
            "outline_width": "0.15" if cls == CLASS_NONE else "0.3",
        })
        cats.append(QgsRendererCategory(cls, sym, CLASS_LABEL[cls]))
    layer.setRenderer(QgsCategorizedSymbolRenderer("class_code", cats))
