"""User-facing wording for eco-driving tips: plain sentences and rounded numbers (no Streamlit code)."""

# feature -> (sentence with {you} and {eco}, multiplier applied to the raw values, number format)
TIP_COMPARISON = {
    "idle_frac": ("You idled for {you}% of the trip; eco drivers on this kind of road idle {eco}%.", 100, "{:.0f}"),
    "stops_per_km": ("You stopped {you} times per km; eco drivers on this kind of road stop {eco}.", 1, "{:.1f}"),
    "hard_accel_per_km": ("You accelerated hard {you} times per km; eco drivers {eco}.", 1, "{:.1f}"),
    "hard_brake_per_km": ("You braked hard {you} times per km; eco drivers {eco}.", 1, "{:.1f}"),
    "rpa": ("You sped up and slowed down more than eco drivers (acceleration effort {you} vs {eco} m/s²).", 1,
            "{:.2f}"),
    "speed_std": ("Your speed varied more than eco drivers' ({you} vs {eco} km/h).", 1, "{:.0f}"),
    "frac_rpm_gt2500": ("Your engine was above 2,500 RPM for {you}% of the trip; eco drivers {eco}%.", 100, "{:.0f}"),
    "rpm_p90": ("Your engine often reached {you} RPM; eco drivers stay around {eco} RPM.", 1, "{:,.0f}"),
    "frac_speed_gt90": ("You drove above 90 km/h for {you}% of the trip; eco drivers {eco}%.", 100, "{:.0f}"),
}


def tip_comparison(tip: dict) -> str:
    """'Your speed varied more than eco drivers' (38 vs 31 km/h).' for a tip from recommend()."""
    sentence, mult, fmt = TIP_COMPARISON.get(tip.get("feature"), ("You were above eco drivers ({you} vs {eco}).", 1,
                                                                  "{:.1f}"))
    you, eco = tip.get("your_value"), tip.get("eco_value")
    if you is None or eco is None:
        return ""
    return sentence.format(you=fmt.format(you * mult), eco=fmt.format(eco * mult))


def saving_text(litres_per_100km) -> str:
    """'about 0.7 L/100 km' (one decimal; '< 0.1 L/100 km' for tiny savings)."""
    if litres_per_100km is None:
        return ""
    return "less than 0.1 L/100 km" if litres_per_100km < 0.05 else f"about {litres_per_100km:.1f} L/100 km"


# what a saving comes from, per tip feature (used in the eco-score card)
SAVING_SOURCE = {
    "idle_frac": "less idling", "stops_per_km": "fewer stops", "hard_accel_per_km": "gentler acceleration",
    "hard_brake_per_km": "earlier, softer braking", "rpa": "a steadier pace", "speed_std": "a more even speed",
    "frac_rpm_gt2500": "earlier gear changes", "rpm_p90": "lower revs",
    "frac_speed_gt90": "a little less speed on the highway",
}


def eco_summary(eco_score, style, tips, eco_l100, saving_pct) -> str:
    """'Smooth driving. Less idling could bring this to about 8.0 L/100 km (up to 21% less).'"""
    parts = []
    if eco_score is not None and eco_score == eco_score:
        if eco_score >= 70:
            parts.append("Smooth driving.")
        elif style == "aggressive" or eco_score < 40:
            parts.append("Frequent hard acceleration or braking.")
    sources = [SAVING_SOURCE[t["feature"]] for t in sorted(tips or [], key=lambda t: -t["saving_l_per_100km"])
               if t.get("feature") in SAVING_SOURCE][:2]
    if eco_l100 is not None and saving_pct is not None and saving_pct >= 1:
        cause = " and ".join(sources) if sources else "driving like eco drivers on this kind of road"
        parts.append(f"{cause[0].upper() + cause[1:]} could bring this to about {eco_l100:.1f} L/100 km "
                     f"(up to {saving_pct:.0f}% less).")
    else:
        parts.append("This trip is already close to eco driving for its road type.")
    return " ".join(parts)
