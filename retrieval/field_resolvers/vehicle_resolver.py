"""
Vehicle Resolver: Resolves motor-specific identifiers: registration number, make, model, engine, chassis.
Uses table grid cells and coordinates first; enforces hard validation against non-values like 'of Vehicle'.
"""

import re
from typing import Dict, Any, List, Optional
from retrieval.field_resolvers.base_resolver import BaseFieldResolver, ResolvedFieldCandidate
from ingestion.table_extractor import find_table_field


class VehicleResolver(BaseFieldResolver):
    field_name: str = "vehicle_details"

    REG_ALIASES = [
        "registration mark & no",
        "registration mark and no",
        "registration mark & number",
        "registration number",
        "registration no",
        "reg no",
        "regn no",
        "vehicle reg no",
        "vehicle registration no",
        "vehicle regn no",
    ]

    ENGINE_ALIASES = [
        "engine number",
        "engine no",
        "engine / motor no",
        "motor no",
    ]

    CHASSIS_ALIASES = [
        "chassis number",
        "chassis no",
        "vin",
    ]

    MAKE_ALIASES = [
        "vehicle make",
        "make of vehicle",
        "make",
        "manufacturer",
    ]

    MODEL_ALIASES = [
        "model/vehicle variant (sub-type)",
        "model/vehicle variant",
        "model / variant",
        "model - variant",
        "model/variant",
        "vehicle model",
        "model",
        "variant",
        "sub-type",
        "vehicle description",
    ]

    DISALLOWED_MODELS = [
        "OF VEHICLE", "VEHICLE", "MAKE", "MODEL", "TYPE OF BODY", "GVW",
        "DETAILS", "DESCRIPTION", "SEATING", "CAPACITY", "NONE", "NA",
        "SUB-TYPE", "VARIANT", "(SUB-", "(SUB-TYPE)", "SUB-", "TYPE)", "TYPE",
        "CHASSIS", "CHASSIS NO", "CHASSIS NO.", "ENGINE", "ENGINE NO", "ENGINE NO.",
        "REGISTRATION", "REGISTRATION NO", "REG NO", "PARTNER", "INVOICE"
    ]

    KNOWN_MAKES = [
        "MARUTI SUZUKI", "MARUTI", "TATA MOTORS", "TATA", "HYUNDAI", "HONDA",
        "MAHINDRA", "BAJAJ", "HERO MOTOCORP", "HERO", "TVS", "ROYAL ENFIELD",
        "YAMAHA", "SUZUKI", "TOYOTA", "KIA", "VOLKSWAGEN", "SKODA", "RENAULT",
        "NISSAN", "FORD", "MG MOTORS", "MG", "EICHER", "ASHOK LEYLAND", "BHARATBENZ"
    ]

    def resolve(
        self,
        full_text: str,
        page_texts: Dict[int, str],
        layout: Optional[Any] = None,
        tables: Optional[List[Dict[str, Any]]] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Optional[ResolvedFieldCandidate]]:
        results: Dict[str, Optional[ResolvedFieldCandidate]] = {
            "registration_number": None,
            "vehicle_make": None,
            "vehicle_model": None,
            "engine_number": None,
            "chassis_number": None,
            "manufacturing_year": None,
            "seating_capacity": None,
            "total_idv": None,
        }

        # -------------------------------------------------------------
        # STEP 1: Table Grid Cell Extraction (Highest Priority)
        # -------------------------------------------------------------
        if tables:
            # 1. Registration Number
            t_reg = find_table_field(
                tables,
                self.REG_ALIASES,
                value_pattern=r"^[A-Z]{2}\s*\d{1,2}\s*[A-Z]{1,3}\s*\d{1,4}$",
                max_down=4,
                max_right=4,
                max_page=2
            )
            if t_reg:
                results["registration_number"] = ResolvedFieldCandidate(
                    field_name="registration_number",
                    value=t_reg["value"],
                    exact_label=t_reg["exact_label"],
                    raw_evidence=t_reg["evidence"],
                    page=t_reg["page"],
                    confidence=0.99,
                    method=t_reg["method"],
                    is_verified=True
                )

            # 2. Make
            t_make = find_table_field(
                tables,
                self.MAKE_ALIASES,
                value_pattern=None,
                max_down=4,
                max_right=4,
                max_page=2
            )
            if t_make and self._is_valid_model(t_make["value"]):
                self._assign_make_model(results, t_make["value"], t_make["exact_label"], t_make["evidence"], t_make["page"], t_make["method"], confidence=0.99)

            # 2b. Model
            if not results["vehicle_model"]:
                t_mod = find_table_field(
                    tables,
                    self.MODEL_ALIASES,
                    value_pattern=None,
                    max_down=4,
                    max_right=4,
                    max_page=2
                )
                if t_mod and self._is_valid_model(t_mod["value"]):
                    self._assign_make_model(results, t_mod["value"], t_mod["exact_label"], t_mod["evidence"], t_mod["page"], t_mod["method"], confidence=0.99)

            # 3. Engine Number
            t_eng = find_table_field(
                tables,
                self.ENGINE_ALIASES,
                value_pattern=r"^[A-Za-z0-9]{5,25}$",
                max_down=4,
                max_right=4,
                max_page=2
            )
            if t_eng and not any(ign in t_eng["value"].upper() for ign in ["SEATING", "CAPACITY", "RC", "MODEL", "CHASSIS", "ENGINE NO", "REGISTRATION", "REG NO"]):
                results["engine_number"] = ResolvedFieldCandidate(
                    field_name="engine_number",
                    value=t_eng["value"],
                    exact_label=t_eng["exact_label"],
                    raw_evidence=t_eng["evidence"],
                    page=t_eng["page"],
                    confidence=0.99,
                    method=t_eng["method"],
                    is_verified=True
                )

            # 4. Chassis Number
            t_cha = find_table_field(
                tables,
                self.CHASSIS_ALIASES,
                value_pattern=r"^[A-Za-z0-9]{5,25}$",
                max_down=4,
                max_right=4,
                max_page=2
            )
            if t_cha and not any(ign in t_cha["value"].upper() for ign in ["SEATING", "CAPACITY", "5", "PACKAGE", "ENGINE", "CHASSIS NO", "REGISTRATION", "REG NO"]):
                results["chassis_number"] = ResolvedFieldCandidate(
                    field_name="chassis_number",
                    value=t_cha["value"],
                    exact_label=t_cha["exact_label"],
                    raw_evidence=t_cha["evidence"],
                    page=t_cha["page"],
                    confidence=0.99,
                    method=t_cha["method"],
                    is_verified=True
                )

            # 5. Manufacturing Year
            t_yr = find_table_field(
                tables,
                ["year of manuf", "year of mfg", "manufacturing year", "mfg year"],
                value_pattern=r"^\d{4}$",
                max_down=4,
                max_right=4,
                max_page=2
            )
            if t_yr:
                results["manufacturing_year"] = ResolvedFieldCandidate(
                    field_name="manufacturing_year",
                    value=t_yr["value"],
                    exact_label=t_yr["exact_label"],
                    raw_evidence=t_yr["evidence"],
                    page=t_yr["page"],
                    confidence=0.99,
                    method=t_yr["method"],
                    is_verified=True
                )

            # 6. Seating Capacity
            t_seat = find_table_field(
                tables,
                ["seating capacity as per rc", "seating capacity", "seating"],
                value_pattern=r"^\d{1,2}$",
                max_down=4,
                max_right=4,
                max_page=2
            )
            if t_seat:
                results["seating_capacity"] = ResolvedFieldCandidate(
                    field_name="seating_capacity",
                    value=t_seat["value"],
                    exact_label=t_seat["exact_label"],
                    raw_evidence=t_seat["evidence"],
                    page=t_seat["page"],
                    confidence=0.99,
                    method=t_seat["method"],
                    is_verified=True
                )

            # 7. Total IDV
            t_idv = find_table_field(
                tables,
                ["idv in rs", "total idv", "vehicle idv"],
                value_pattern=r"^\d+(?:\.\d{2})?$",
                max_down=4,
                max_right=4,
                max_page=2
            )
            if t_idv:
                results["total_idv"] = ResolvedFieldCandidate(
                    field_name="total_idv",
                    value=t_idv["value"],
                    exact_label=t_idv["exact_label"],
                    raw_evidence=t_idv["evidence"],
                    page=t_idv["page"],
                    confidence=0.99,
                    method=t_idv["method"],
                    is_verified=True
                )

        # -------------------------------------------------------------
        # STEP 2: Spatial Coordinate Extraction via Layout
        # -------------------------------------------------------------
        if layout and hasattr(layout, "extract_value_for_label"):
            if not results["registration_number"]:
                s_regs = layout.extract_value_for_label(
                    label_aliases=self.REG_ALIASES,
                    value_pattern=r"^[A-Z]{2}\s*\d{1,2}\s*[A-Z]{1,3}\s*\d{1,4}$"
                )
                if s_regs:
                    sc = s_regs[0]
                    results["registration_number"] = ResolvedFieldCandidate(
                        field_name="registration_number",
                        value=sc["value"],
                        exact_label=sc.get("raw_text", ""),
                        raw_evidence=sc["raw_text"],
                        page=sc["page"],
                        bbox=sc.get("bbox"),
                        confidence=0.95,
                        method=sc.get("method", "spatial"),
                        is_verified=True
                    )

            # 2. Vehicle Make
            if not results["vehicle_make"]:
                s_makes = layout.extract_value_for_label(label_aliases=self.MAKE_ALIASES)
                for sc in s_makes:
                    val = " ".join(sc["value"].strip().split())
                    if val.upper() not in self.DISALLOWED_MODELS and len(val) >= 2:
                        results["vehicle_make"] = ResolvedFieldCandidate(
                            field_name="vehicle_make",
                            value=val,
                            exact_label=sc.get("raw_text", ""),
                            raw_evidence=sc["raw_text"],
                            page=sc["page"],
                            bbox=sc.get("bbox"),
                            confidence=0.96,
                            method=sc.get("method", "spatial"),
                            is_verified=True
                        )
                        break

            # 3. Vehicle Model
            if not results["vehicle_model"]:
                s_models = layout.extract_value_for_label(label_aliases=self.MODEL_ALIASES)
                for sc in s_models:
                    if self._is_valid_model(sc["value"]):
                        results["vehicle_model"] = ResolvedFieldCandidate(
                            field_name="vehicle_model",
                            value=sc["value"],
                            exact_label=sc.get("raw_text", ""),
                            raw_evidence=sc["raw_text"],
                            page=sc["page"],
                            bbox=sc.get("bbox"),
                            confidence=0.96,
                            method=sc.get("method", "spatial"),
                            is_verified=True
                        )
                        break

            # 4. Combined Make & Model Fallback
            if not results["vehicle_model"] or not results["vehicle_make"]:
                s_mms = layout.extract_value_for_label(label_aliases=self.MAKE_ALIASES + self.MODEL_ALIASES)
                for sc in s_mms:
                    if self._is_valid_model(sc["value"]):
                        self._assign_make_model(results, sc["value"], sc.get("raw_text", ""), sc["raw_text"], sc["page"], sc.get("method", "spatial"), bbox=sc.get("bbox"), confidence=0.95)
                        break

            if not results["engine_number"]:
                s_engs = layout.extract_value_for_label(
                    label_aliases=self.ENGINE_ALIASES,
                    value_pattern=r"^[A-Za-z0-9]{5,25}$"
                )
                if s_engs:
                    sc = s_engs[0]
                    if not any(ign in sc["value"].upper() for ign in ["SEATING", "CAPACITY", "RC", "CHASSIS", "ENGINE NO", "REGISTRATION", "REG NO"]):
                        results["engine_number"] = ResolvedFieldCandidate(
                            field_name="engine_number",
                            value=sc["value"],
                            exact_label=sc.get("raw_text", ""),
                            raw_evidence=sc["raw_text"],
                            page=sc["page"],
                            bbox=sc.get("bbox"),
                            confidence=0.95,
                            method=sc.get("method", "spatial"),
                            is_verified=True
                        )

            if not results["chassis_number"]:
                s_chas = layout.extract_value_for_label(
                    label_aliases=self.CHASSIS_ALIASES,
                    value_pattern=r"^[A-Za-z0-9]{5,25}$"
                )
                if s_chas:
                    sc = s_chas[0]
                    if not any(ign in sc["value"].upper() for ign in ["SEATING", "CAPACITY", "5", "PACKAGE", "ENGINE", "CHASSIS NO", "REGISTRATION", "REG NO"]):
                        results["chassis_number"] = ResolvedFieldCandidate(
                            field_name="chassis_number",
                            value=sc["value"],
                            exact_label=sc.get("raw_text", ""),
                            raw_evidence=sc["raw_text"],
                            page=sc["page"],
                            bbox=sc.get("bbox"),
                            confidence=0.95,
                            method=sc.get("method", "spatial"),
                            is_verified=True
                        )

        # -------------------------------------------------------------
        # STEP 3: Fallback Label-Adjacent Text Extraction (Pages 1-2 only)
        # -------------------------------------------------------------
        for pno in sorted(page_texts.keys())[:2]:
            text = page_texts[pno]

            # Registration fallback
            if not results["registration_number"]:
                for alias in self.REG_ALIASES:
                    pattern = rf"(?i)\b{re.escape(alias)}\b[\s\.:/#=\-]*([A-Z]{{2}}\s*\d{{1,2}}\s*[A-Z]{{1,3}}\s*\d{{1,4}})"
                    m = re.search(pattern, text)
                    if m:
                        results["registration_number"] = ResolvedFieldCandidate(
                            field_name="registration_number",
                            value=m.group(1).strip(),
                            exact_label=alias,
                            raw_evidence=m.group(0).strip(),
                            page=pno,
                            confidence=0.92,
                            method="reg_alias_match",
                            is_verified=True
                        )
                        break

            # Engine fallback
            if not results["engine_number"]:
                for alias in self.ENGINE_ALIASES:
                    pattern = rf"(?i)\b{re.escape(alias)}\b[\s\.:/#=\-]*([A-Za-z0-9]{{5,25}})"
                    m = re.search(pattern, text)
                    if m:
                        val = m.group(1).strip()
                        if not any(ign in val.upper() for ign in ["SEATING", "CAPACITY", "RC", "CHASSIS", "ENGINE NO", "REGISTRATION", "REG NO"]):
                            results["engine_number"] = ResolvedFieldCandidate(
                                field_name="engine_number",
                                value=val,
                                exact_label=alias,
                                raw_evidence=m.group(0).strip(),
                                page=pno,
                                confidence=0.90,
                                method="engine_alias_match",
                                is_verified=True
                            )
                            break

            # Chassis fallback
            if not results["chassis_number"]:
                for alias in self.CHASSIS_ALIASES:
                    pattern = rf"(?i)\b{re.escape(alias)}\b[\s\.:/#=\-]*([A-Za-z0-9]{{5,25}})"
                    m = re.search(pattern, text)
                    if m:
                        val = m.group(1).strip()
                        if not any(ign in val.upper() for ign in ["SEATING", "CAPACITY", "5", "PACKAGE", "ENGINE", "CHASSIS NO", "REGISTRATION", "REG NO"]):
                            results["chassis_number"] = ResolvedFieldCandidate(
                                field_name="chassis_number",
                                value=val,
                                exact_label=alias,
                                raw_evidence=m.group(0).strip(),
                                page=pno,
                                confidence=0.90,
                                method="chassis_alias_match",
                                is_verified=True
                            )
                            break

            # Make & Model fallback
            if not results["vehicle_model"]:
                for alias in self.MODEL_ALIASES:
                    pattern = rf"(?i)\b{re.escape(alias)}\b[\s\.:/#=\-]*([^\n\r,;]{{3,50}})"
                    m = re.search(pattern, text)
                    if m:
                        raw_val = m.group(1).strip()
                        if self._is_valid_model(raw_val):
                            self._assign_make_model(results, raw_val, alias, m.group(0).strip(), pno, "model_alias_match", confidence=0.90)
                            break

        return results

    def _is_valid_model(self, text: Optional[str]) -> bool:
        if not text:
            return False
        clean = " ".join(text.strip().split()).upper()
        if len(clean) < 3:
            return False
        if clean in self.DISALLOWED_MODELS:
            return False
        if clean.startswith("OF VEHICLE") or clean.startswith("AND MODEL") or clean.startswith("(") or clean.endswith("-"):
            return False
        if any(bad in clean for bad in [
            "LIMITATION", "CONDITIONS", "SCHEDULE", "PAYABLE", "POLICY",
            "SUB-TYPE", "VARIANT", "PERIOD INSURANCE", "PERIOD OF", "FROM 10", "TO 09",
            "CHASSIS", "ENGINE", "REGISTRATION", "SEATING", "CAPACITY", "FASTAG", "ODOMETER",
            "INVOICE", "PARTNER", "PREMIUM", "OF VEHICLE", "TYPE OF"
        ]):
            return False
        return True

    def _assign_make_model(
        self,
        results: Dict[str, Optional[ResolvedFieldCandidate]],
        raw_val: str,
        exact_label: str,
        evidence: str,
        page: int,
        method: str,
        bbox: Optional[Any] = None,
        confidence: float = 0.99
    ):
        """
        Parses raw model string (e.g. 'MARUTI ALTO 800 LXI' or 'TVS JUPITER-ZX' or 'HONDA').
        Extracts vehicle_make and vehicle_model accurately while keeping full string in metadata.
        """
        clean_val = " ".join(raw_val.strip().split())
        upper_val = clean_val.upper()

        detected_make = None
        for mk in self.KNOWN_MAKES:
            if upper_val == mk or upper_val.startswith(mk + " ") or upper_val.startswith(mk + "-") or upper_val.startswith(mk + "/"):
                detected_make = mk
                break
            elif f" {mk} " in f" {upper_val} ":
                detected_make = mk

        model_val = None
        if detected_make and clean_val.upper().startswith(detected_make):
            remainder = clean_val[len(detected_make):].strip(" -/")
            if remainder and len(remainder) >= 2:
                model_val = remainder
        elif not detected_make:
            model_val = clean_val

        # If the label was explicitly a MODEL label, treat clean_val as the model
        is_explicit_model_label = any(m_alias in exact_label.lower() for m_alias in ["model", "variant", "sub-type", "description"])
        if is_explicit_model_label and not model_val:
            model_val = clean_val

        # Set vehicle_make if detected and not already set
        if detected_make and not results.get("vehicle_make"):
            results["vehicle_make"] = ResolvedFieldCandidate(
                field_name="vehicle_make",
                value=detected_make,
                exact_label=exact_label,
                raw_evidence=evidence,
                page=page,
                bbox=bbox,
                confidence=confidence,
                method=method,
                is_verified=True
            )

        # Set vehicle_model only if valid model string found
        if model_val and self._is_valid_model(model_val) and not results.get("vehicle_model"):
            results["vehicle_model"] = ResolvedFieldCandidate(
                field_name="vehicle_model",
                value=model_val,
                exact_label=exact_label,
                raw_evidence=evidence,
                page=page,
                bbox=bbox,
                confidence=confidence,
                method=method,
                is_verified=True,
                metadata={"full_model_str": clean_val, "combined_make_model": f"{detected_make} {model_val}".strip() if detected_make else clean_val}
            )
