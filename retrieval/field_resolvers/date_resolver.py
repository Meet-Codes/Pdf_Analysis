"""
Date Resolver: High-precision date extraction, interval validation, and duration calculation.
Rule: Never take 'first two dates', never take 'min/max'. Strictly parse verified pairs/labels.
"""

import re
from datetime import datetime, date
from typing import Dict, Any, List, Optional, Tuple
from retrieval.field_resolvers.base_resolver import BaseFieldResolver, ResolvedFieldCandidate


class DateResolver(BaseFieldResolver):
    field_name: str = "policy_dates"

    START_ALIASES = [
        "policy start date",
        "start date",
        "inception date",
        "commencement date",
        "effective from",
        "valid from",
        "period from",
        "from date",
        "from:",
        "from",
        "bill date",
    ]

    END_ALIASES = [
        "policy end date",
        "expiry date",
        "expiration date",
        "valid until",
        "valid to",
        "period to",
        "to date",
        "to midnight on",
        "to midnight of",
        "to midnight",
        "expires on",
        "to:",
        "to",
        "due date",
        "payment due date",
        "pay by date",
    ]

    BOOKING_ALIASES = [
        "policy booking date",
        "policy issue date",
        "booking date",
        "issuance date",
        "issue date",
        "date of issue",
        "transaction date",
    ]

    RANGE_PATTERNS = [
        # Explicit Period of Insurance / Policy Period / Period of Policy From ... To ...
        r"(?i)(?:period\s*of\s*insurance|policy\s*period|period\s*of\s*cover|period\s*of\s*policy|policy\s*term)[\s\S]{0,120}?\bfrom[\s\.:]*(\d{1,2}[-/\.]\d{1,2}[-/\.]\d{2,4}|\d{1,2}\s+[A-Za-z]+,?\s+\d{2,4}|\d{1,2}-[A-Za-z]+-\d{2,4})[\s\S]{1,100}?\bto\b[\s\.:]*(?:midnight\s+(?:on|of)\s+)?(\d{1,2}[-/\.]\d{1,2}[-/\.]\d{2,4}|\d{1,2}\s+[A-Za-z]+,?\s+\d{2,4}|\d{1,2}-[A-Za-z]+-\d{2,4})",
        # Generic From ... To ...
        r"(?i)\bfrom[\s\.:]*(\d{1,2}[-/\.]\d{1,2}[-/\.]\d{2,4}|\d{1,2}\s+[A-Za-z]+,?\s+\d{2,4}|\d{1,2}-[A-Za-z]+-\d{2,4})[\s\w\.:,/]*(?:hrs|hours|midnight)?[\s\S]{1,60}?\bto\b[\s\.:]*(?:midnight\s+(?:on|of)\s+)?(\d{1,2}[-/\.]\d{1,2}[-/\.]\d{2,4}|\d{1,2}\s+[A-Za-z]+,?\s+\d{2,4}|\d{1,2}-[A-Za-z]+-\d{2,4})",
    ]

    def resolve(
        self,
        full_text: str,
        page_texts: Dict[int, str],
        layout: Optional[Any] = None,
        tables: Optional[List[Dict[str, Any]]] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Optional[ResolvedFieldCandidate]]:
        """
        Resolves start date, end date, and computed duration.
        Returns:
        {
            "policy_start_date": ResolvedFieldCandidate or None,
            "policy_end_date": ResolvedFieldCandidate or None,
            "policy_duration": ResolvedFieldCandidate or None,
            "policy_booking_date": ResolvedFieldCandidate or None
        }
        """
        results: Dict[str, Optional[ResolvedFieldCandidate]] = {
            "policy_start_date": None,
            "policy_end_date": None,
            "policy_duration": None,
            "policy_booking_date": None,
        }

        # Step 1: Detect explicit range pairs from text (Highest priority & confidence)
        for pno, text in page_texts.items():
            for pat in self.RANGE_PATTERNS:
                m = re.search(pat, text)
                if m:
                    raw_s, raw_e = m.group(1).strip(), m.group(2).strip()
                    dt_s = self._parse_date_safe(raw_s)
                    dt_e = self._parse_date_safe(raw_e)

                    if dt_s and dt_e and dt_s <= dt_e:
                        results["policy_start_date"] = ResolvedFieldCandidate(
                            field_name="policy_start_date",
                            value=raw_s,
                            raw_evidence=m.group(0).strip(),
                            page=pno,
                            confidence=0.99,
                            method="range_pattern",
                            is_verified=True
                        )
                        results["policy_end_date"] = ResolvedFieldCandidate(
                            field_name="policy_end_date",
                            value=raw_e,
                            raw_evidence=m.group(0).strip(),
                            page=pno,
                            confidence=0.99,
                            method="range_pattern",
                            is_verified=True
                        )
                        duration_str = self._calculate_duration(dt_s, dt_e)
                        results["policy_duration"] = ResolvedFieldCandidate(
                            field_name="policy_duration",
                            value=duration_str,
                            raw_evidence=f"Computed from {raw_s} to {raw_e}",
                            page=pno,
                            confidence=0.99,
                            method="computed_interval",
                            is_verified=True
                        )
                        break
            if results["policy_start_date"]:
                break

        # Step 2: Spatial coordinate extraction via layout
        if layout and hasattr(layout, "extract_value_for_label"):
            if not results["policy_start_date"]:
                s_cands = layout.extract_value_for_label(
                    label_aliases=self.START_ALIASES,
                    value_pattern=r"(\d{1,2}[-/\.]\d{1,2}[-/\.]\d{2,4}|\d{1,2}\s+[A-Za-z]+,?\s+\d{2,4}|\d{1,2}-[A-Za-z]+-\d{2,4})"
                )
                for sc in s_cands:
                    dt = self._parse_date_safe(sc["value"])
                    if dt:
                        results["policy_start_date"] = ResolvedFieldCandidate(
                            field_name="policy_start_date",
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

            if not results["policy_end_date"]:
                e_cands = layout.extract_value_for_label(
                    label_aliases=self.END_ALIASES,
                    value_pattern=r"(\d{1,2}[-/\.]\d{1,2}[-/\.]\d{2,4}|\d{1,2}\s+[A-Za-z]+,?\s+\d{2,4}|\d{1,2}-[A-Za-z]+-\d{2,4})"
                )
                for sc in e_cands:
                    dt = self._parse_date_safe(sc["value"])
                    if dt:
                        results["policy_end_date"] = ResolvedFieldCandidate(
                            field_name="policy_end_date",
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

            if not results["policy_booking_date"]:
                b_cands = layout.extract_value_for_label(
                    label_aliases=self.BOOKING_ALIASES,
                    value_pattern=r"(\d{1,2}[-/\.]\d{1,2}[-/\.]\d{2,4}|\d{1,2}\s+[A-Za-z]+,?\s+\d{2,4}|\d{1,2}-[A-Za-z]+-\d{2,4})"
                )
                for sc in b_cands:
                    dt = self._parse_date_safe(sc["value"])
                    if dt:
                        results["policy_booking_date"] = ResolvedFieldCandidate(
                            field_name="policy_booking_date",
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

        # Step 3: Individual label regex extraction
        if not results["policy_start_date"]:
            cand_start = self._extract_single_date(page_texts, self.START_ALIASES, "policy_start_date")
            results["policy_start_date"] = cand_start

        if not results["policy_end_date"]:
            cand_end = self._extract_single_date(page_texts, self.END_ALIASES, "policy_end_date")
            results["policy_end_date"] = cand_end

        if not results["policy_booking_date"]:
            cand_book = self._extract_single_date(page_texts, self.BOOKING_ALIASES, "policy_booking_date")
            results["policy_booking_date"] = cand_book

        # Compute duration if start and end are present
        if results["policy_start_date"] and results["policy_end_date"] and not results["policy_duration"]:
            dt_s = self._parse_date_safe(str(results["policy_start_date"].value))
            dt_e = self._parse_date_safe(str(results["policy_end_date"].value))
            if dt_s and dt_e and dt_s <= dt_e:
                results["policy_duration"] = ResolvedFieldCandidate(
                    field_name="policy_duration",
                    value=self._calculate_duration(dt_s, dt_e),
                    raw_evidence=f"Computed from {results['policy_start_date'].value} to {results['policy_end_date'].value}",
                    page=results["policy_start_date"].page,
                    confidence=0.95,
                    method="computed_interval",
                    is_verified=True
                )

        return results

    def _extract_single_date(
        self,
        page_texts: Dict[int, str],
        aliases: List[str],
        field_name: str
    ) -> Optional[ResolvedFieldCandidate]:
        for pno, text in page_texts.items():
            for alias in aliases:
                # Use clean boundary matching that works with punctuation
                esc = re.escape(alias)
                pattern = rf"(?i)(?:^|[\s\n\r])({esc})[\s\.:/#=\-]*(?:midnight\s+(?:on|of)\s+)?(\d{{1,2}}[-/\.]\d{{1,2}}[-/\.]\d{{2,4}}|\d{{1,2}}\s+[A-Za-z]+,?\s+\d{{2,4}}|\d{{1,2}}-[A-Za-z]+-\d{{2,4}})"
                m = re.search(pattern, text)
                if m:
                    raw_val = m.group(2).strip()
                    dt = self._parse_date_safe(raw_val)
                    if dt:
                        return ResolvedFieldCandidate(
                            field_name=field_name,
                            value=raw_val,
                            raw_evidence=m.group(0).strip(),
                            page=pno,
                            confidence=0.92,
                            method="single_label_regex",
                            is_verified=True
                        )
        return None

    def _parse_date_safe(self, date_str: str) -> Optional[date]:
        if not date_str:
            return None
        # Clean string: strip trailing time, midnight, hours
        cleaned = re.sub(r"(?i)\s+(?:midnight|hrs|hours).*$", "", str(date_str).strip())
        cleaned = re.sub(r"\s+\d{1,2}:\d{2}(?::\d{2})?", "", cleaned.strip())
        cleaned = cleaned.replace(",", "").strip()

        from dateutil import parser as dp
        try:
            dt = dp.parse(cleaned, dayfirst=True)
            if 1950 <= dt.year <= 2100:
                return dt.date()
        except Exception:
            pass

        clean_slash = cleaned.replace(".", "/").replace("-", "/")
        for fmt in ["%d/%m/%Y", "%Y/%m/%d", "%d/%m/%y"]:
            try:
                return datetime.strptime(clean_slash, fmt).date()
            except ValueError:
                continue
        for fmt in ["%d %B %Y", "%d %b %Y"]:
            try:
                return datetime.strptime(cleaned, fmt).date()
            except ValueError:
                continue
        return None

    def _calculate_duration(self, start: date, end: date) -> str:
        days = (end - start).days
        if 350 <= days <= 370:
            return "1 Year"
        elif 710 <= days <= 740:
            return "2 Years"
        elif 1080 <= days <= 1110:
            return "3 Years"
        elif 1800 <= days <= 1850:
            return "5 Years"
        elif 25 <= days <= 35:
            return "1 Month"
        return f"{days} Days"
