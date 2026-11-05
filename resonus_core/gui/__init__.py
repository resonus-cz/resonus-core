"""GTK 4 / libadwaita host for resonus apps (GNOME HIG).

Import this package only in GUI code: it requires PyGObject, GTK 4 and libadwaita.
"""

import gi

gi.require_version("Gtk", "4.0")
gi.require_version("Adw", "1")
