class PipelineConfig:
    
    # Transaction dataset columns (primary source of truth)
    TRANSACTION_COLUMNS = {
        'item': 'Item',
        'vgn': 'VGN',
        'vpn': 'VPN',
        
        # Description columns
        'item_desc_1': 'Item Desc 1',
        'item_desc_2': 'Item Desc 2',
        'class_description': 'Class Description',
        'div_description': 'Div Description',
        
        # Category columns
        'item_category': 'Item Category',
        'item_sub_category': 'Item Sub Category',
        
        # Location columns
        'region': 'Region',
        'territory': 'Territory',
        'location_ps': 'Location PS',
        'dashboard_location': 'Dashboard Location',
        'warehouse': 'Warehouse',
        'whs_code': 'Whs Code',
        
        # System columns
        'erp_system': 'ERP System',
        'selling_uom': 'Selling UOM',
        'im_uom': 'IM UoM',
        
        # Cost and quantity columns
        'qty': 'Qty',
        'gross_cost': 'Gross Cost',
        'net_cost': 'Net Cost',
        'case_pack': 'Case Pack',
        
        # Vendor and customer columns
        'preferred_vendor_code': 'Preferred Vendor Code',
        'preferred_vendor_name': 'Preferred Vendor Name',
        'customer_code': 'Customer Code',
        'customer_class': 'Customer Class',
        'national_account_customer': 'National Account Customer',
        'price_group_name': 'Price Group Name',
        'updated_salesperson_name': 'Updated Salesperson Name',
        
        # Flag columns
        'vb_flag': 'VB Flag',
        'network_flag': 'Network Flag',
        'pod': 'POD'
    }