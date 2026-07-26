# AAAI Course-Reward Readability Design

## Goal

Improve the readability of the `Course-info Reward` panel by enlarging its semantic icons and internal typography without changing the panel or chip geometry.

## Change

- Increase the course book icon from `role_scale(30, "icon")` to `role_scale(40, "icon")` and its line width from 1.1 to 1.3.
- Increase all four reward-term icons from `role_scale(15, "icon")` to `role_scale(20, "icon")`.
- Increase the panel heading from `role_scale(9.2, "component")` to `role_scale(10.0, "component")`.
- Increase the course symbol from `role_scale(8.2, "symbol")` to `role_scale(9.0, "symbol")`.
- Increase the four reward labels from `role_scale(7.0, "symbol")` to `role_scale(8.0, "symbol")`.
- Preserve all existing panel, chip, and text anchor coordinates.

## Verification

- Assert the book-icon size is 40 and every reward-term icon size is 20 before role scaling.
- Assert the panel title, course symbol, and chip labels use the new font-size values.
- Confirm all four reward labels remain within their chip borders and do not collide with the enlarged icons.
- Confirm every enlarged icon remains inside its panel or chip boundary.
- Regenerate SVG, PDF, PNG, and TIFF outputs and compile the paper without fatal errors, overfull boxes, or undefined references.
