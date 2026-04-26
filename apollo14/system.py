from dataclasses import dataclass, field

from apollo14.materials import Material, air


@dataclass
class OpticalSystem:
    """Container for optical elements + environment material."""
    elements: list = field(default_factory=list)
    env_material: Material = field(default_factory=lambda: air)

    def add(self, element):
        self.elements.append(element)

    def resolve(self, name):
        """Look up an element by name.

        ``name`` is either a plain string (``"mirror_0"``, ``"pupil"``) or a
        tuple for addressable sub-elements (``("chassis", "back")`` → the
        named face of the GlassBlock).

        The returned object exposes ``to_generic_surface(current_material, mode)``.
        """
        if isinstance(name, tuple):
            block_name, sub_name = name
            for e in self.elements:
                if getattr(e, "name", None) == block_name and hasattr(e, "get_face"):
                    return e.get_face(sub_name)
            available = [getattr(e, "name", "?") for e in self.elements
                         if hasattr(e, "get_face")]
            raise KeyError(
                f"No block named '{block_name}' in system. Available: {available}")

        for e in self.elements:
            if getattr(e, "name", None) == name:
                return e
        available = [getattr(e, "name", "?") for e in self.elements]
        raise KeyError(f"No element named '{name}'. Available: {available}")
