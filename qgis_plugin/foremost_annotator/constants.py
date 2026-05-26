CLASS_NONE = 0
CLASS_HAB  = 1
CLASS_RA   = 2
CLASS_NR   = 3

CLASS_LABEL = {
    CLASS_NONE: "Unlabeled",
    CLASS_HAB:  "Habitat",
    CLASS_RA:   "Restorable",
    CLASS_NR:   "Non-Restorable",
}

# RGBA fill colours
CLASS_FILL = {
    CLASS_NONE: "255,255,255,0",
    CLASS_HAB:  "45,106,79,160",
    CLASS_RA:   "244,162,97,160",
    CLASS_NR:   "173,181,189,130",
}
CLASS_STROKE = {
    CLASS_NONE: "#aaaaaa",
    CLASS_HAB:  "#1b4332",
    CLASS_RA:   "#e76f51",
    CLASS_NR:   "#6c757d",
}

# Vector layer field indices (must match GridManager.FIELDS order)
FLD_ROW   = 0
FLD_COL   = 1
FLD_CLASS = 2
FLD_COST  = 3
