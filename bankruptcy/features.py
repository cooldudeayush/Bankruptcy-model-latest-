"""The 64 financial ratios in the Polish Companies Bankruptcy dataset.

Each entry is  AttrN: (official UCI definition, short plain-English meaning).
"""

FEATURES = {
    "Attr1": ("Net profit / total assets", "Return on assets: profit made per unit of assets"),
    "Attr2": ("Total liabilities / total assets", "Debt ratio: share of assets financed by debt"),
    "Attr3": ("Working capital / total assets", "Short-term cushion relative to company size"),
    "Attr4": ("Current assets / short-term liabilities", "Current ratio: can it pay bills due within a year?"),
    "Attr5": ("[(Cash + short-term securities + receivables - short-term liabilities) / (operating expenses - depreciation)] * 365",
              "Days the company could keep running on its liquid assets"),
    "Attr6": ("Retained earnings / total assets", "Profits kept in the business over its life"),
    "Attr7": ("EBIT / total assets", "Operating return on assets (before interest and tax)"),
    "Attr8": ("Book value of equity / total liabilities", "Owners' money compared with debt"),
    "Attr9": ("Sales / total assets", "Asset turnover: sales generated per unit of assets"),
    "Attr10": ("Equity / total assets", "Share of assets owned outright by shareholders"),
    "Attr11": ("(Gross profit + extraordinary items + financial expenses) / total assets", "Broad profit measure relative to assets"),
    "Attr12": ("Gross profit / short-term liabilities", "Profit available to cover short-term debt"),
    "Attr13": ("(Gross profit + depreciation) / sales", "Cash-style profit margin"),
    "Attr14": ("(Gross profit + interest) / total assets", "Profit before interest relative to assets"),
    "Attr15": ("(Total liabilities * 365) / (gross profit + depreciation)", "Days of profit needed to repay all debt"),
    "Attr16": ("(Gross profit + depreciation) / total liabilities", "Cash-style profit relative to debt"),
    "Attr17": ("Total assets / total liabilities", "Assets available per unit of debt"),
    "Attr18": ("Gross profit / total assets", "Gross return on assets"),
    "Attr19": ("Gross profit / sales", "Gross profit margin"),
    "Attr20": ("(Inventory * 365) / sales", "Days of sales sitting in inventory"),
    "Attr21": ("Sales (n) / sales (n-1)", "Sales growth versus last year"),
    "Attr22": ("Profit on operating activities / total assets", "Operating return on assets"),
    "Attr23": ("Net profit / sales", "Net profit margin"),
    "Attr24": ("Gross profit (in 3 years) / total assets", "Three-year profit relative to assets"),
    "Attr25": ("(Equity - share capital) / total assets", "Accumulated reserves relative to assets"),
    "Attr26": ("(Net profit + depreciation) / total liabilities", "Cash flow relative to debt"),
    "Attr27": ("Profit on operating activities / financial expenses", "Interest cover: operating profit per unit of finance costs"),
    "Attr28": ("Working capital / fixed assets", "Short-term cushion relative to long-term assets"),
    "Attr29": ("Logarithm of total assets", "Company size"),
    "Attr30": ("(Total liabilities - cash) / sales", "Net debt relative to sales"),
    "Attr31": ("(Gross profit + interest) / sales", "Margin before interest"),
    "Attr32": ("(Current liabilities * 365) / cost of products sold", "Days taken to pay suppliers"),
    "Attr33": ("Operating expenses / short-term liabilities", "Running costs relative to short-term debt"),
    "Attr34": ("Operating expenses / total liabilities", "Running costs relative to total debt"),
    "Attr35": ("Profit on sales / total assets", "Sales profit relative to assets"),
    "Attr36": ("Total sales / total assets", "Total asset turnover"),
    "Attr37": ("(Current assets - inventories) / long-term liabilities", "Liquid assets relative to long-term debt"),
    "Attr38": ("Constant capital / total assets", "Long-term funding relative to assets"),
    "Attr39": ("Profit on sales / sales", "Sales profit margin"),
    "Attr40": ("(Current assets - inventory - receivables) / short-term liabilities", "Cash ratio: cash-like assets vs short-term debt"),
    "Attr41": ("Total liabilities / ((profit on operating activities + depreciation) * (12/365))", "How long operating cash flow would take to repay debt"),
    "Attr42": ("Profit on operating activities / sales", "Operating profit margin"),
    "Attr43": ("Rotation receivables + inventory turnover in days", "Days to turn stock and receivables into cash"),
    "Attr44": ("(Receivables * 365) / sales", "Days customers take to pay"),
    "Attr45": ("Net profit / inventory", "Profit relative to stock held"),
    "Attr46": ("(Current assets - inventory) / short-term liabilities", "Quick ratio: liquid assets vs short-term debt"),
    "Attr47": ("(Inventory * 365) / cost of products sold", "Days to sell inventory"),
    "Attr48": ("EBITDA (profit on operating activities - depreciation) / total assets", "EBITDA relative to assets"),
    "Attr49": ("EBITDA (profit on operating activities - depreciation) / sales", "EBITDA margin"),
    "Attr50": ("Current assets / total liabilities", "Current assets relative to all debt"),
    "Attr51": ("Short-term liabilities / total assets", "Share of assets financed by short-term debt"),
    "Attr52": ("(Short-term liabilities * 365) / cost of products sold", "Days of costs covered by short-term debt"),
    "Attr53": ("Equity / fixed assets", "Owners' money relative to long-term assets"),
    "Attr54": ("Constant capital / fixed assets", "Long-term funding relative to long-term assets"),
    "Attr55": ("Working capital", "Current assets minus short-term liabilities (absolute amount)"),
    "Attr56": ("(Sales - cost of products sold) / sales", "Margin after direct costs"),
    "Attr57": ("(Current assets - inventory - short-term liabilities) / (sales - gross profit - depreciation)", "Liquid cushion relative to cash costs"),
    "Attr58": ("Total costs / total sales", "Cost ratio: costs per unit of sales (above 1 means losing money)"),
    "Attr59": ("Long-term liabilities / equity", "Long-term debt relative to owners' money"),
    "Attr60": ("Sales / inventory", "Inventory turnover"),
    "Attr61": ("Sales / receivables", "Receivables turnover"),
    "Attr62": ("(Short-term liabilities * 365) / sales", "Days of sales needed to cover short-term debt"),
    "Attr63": ("Sales / short-term liabilities", "Sales relative to short-term debt"),
    "Attr64": ("Sales / fixed assets", "Fixed-asset turnover"),
}

RATIO_COLUMNS = list(FEATURES)  # Attr1 ... Attr64, in dataset order


def definition(name: str) -> str:
    return FEATURES[name][0] if name in FEATURES else name


def meaning(name: str) -> str:
    return FEATURES[name][1] if name in FEATURES else name
