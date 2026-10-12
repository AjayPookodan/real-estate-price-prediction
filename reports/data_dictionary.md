# Data Dictionary

## Dataset Overview

- **Dataset file:** `data/raw/housing_raw.csv`
- **Number of rows:** 10,704
- **Number of columns:** 67
- **Purpose:** Document the fields in the raw real-estate dataset.
- **Note:** Column meanings and units marked for verification must be
  confirmed against the original dataset documentation.

## Column Descriptions

| Column                       | Meaning                                             | Type                | Unit                                   | Missing %   | Notes                                                                                    |
|:-----------------------------|:----------------------------------------------------|:--------------------|:---------------------------------------|:------------|:-----------------------------------------------------------------------------------------|
| PROP_ID                      | Unique property identifier                          | identifier          | Not applicable                         | 0.00%       | Not normally used as a prediction feature; pandas dtype: str                             |
| PHOTO_URL                    | URL for the property's photo                        | text/URL            | Not applicable                         | 17.19%      | Image reference, not a numeric feature; pandas dtype: str                                |
| MEDIUM_PHOTO_URL             | URL for a medium-sized property photo               | text/URL            | Not applicable                         | 17.19%      | Image reference; pandas dtype: str                                                       |
| PREFERENCE                   | Listing preference or category                      | categorical         | Not applicable                         | 0.00%       | Meaning should be verified from source documentation; pandas dtype: str                  |
| DESCRIPTION                  | Textual description of the property                 | text                | Not applicable                         | 0.00%       | Unstructured text; pandas dtype: str                                                     |
| PROPERTY_TYPE                | Type or category of property                        | categorical         | Not applicable                         | 0.00%       | Potential prediction feature; pandas dtype: str                                          |
| CITY                         | City associated with the property                   | categorical         | Not applicable                         | 0.00%       | Verify geographic coverage from actual values; pandas dtype: str                         |
| LOCALITY                     | Locality or neighbourhood of the property           | categorical/text    | Not applicable                         | 0.00%       | May help represent location; pandas dtype: str                                           |
| TRANSACT_TYPE                | Transaction category                                | categorical         | Not applicable                         | 5.16%       | Inspect values to distinguish sale, rent, and other listings; pandas dtype: float64      |
| OWNTYPE                      | Property ownership category                         | categorical         | Not applicable                         | 0.00%       | Verify category definitions; pandas dtype: int64                                         |
| BEDROOM_NUM                  | Number of bedrooms                                  | numeric             | count                                  | 1.80%       | Potential prediction feature; pandas dtype: float64                                      |
| BATHROOM_NUM                 | Number of bathrooms                                 | numeric             | count                                  | 0.00%       | Potential prediction feature if present; pandas dtype: int64                             |
| BALCONY_NUM                  | Number of balconies                                 | numeric             | count                                  | 1.90%       | Potential prediction feature if present; pandas dtype: float64                           |
| PRICE_PER_UNIT_AREA          | Price per unit of property area                     | text/numeric        | Verify from source                     | 0.00%       | Check units before using; pandas dtype: float64                                          |
| FURNISH                      | Furnishing category                                 | categorical         | Not applicable                         | 0.00%       | Verify category values; pandas dtype: int64                                              |
| FACING                       | Direction the property faces                        | categorical         | Not applicable                         | 0.00%       | Potential prediction feature; pandas dtype: int64                                        |
| AGE                          | Property age or age category                        | numeric/categorical | Years or category; verify from source  | 0.00%       | Inspect format before converting; pandas dtype: int64                                    |
| FLOOR_NUM                    | Floor on which the property is located              | numeric/text        | floor number                           | 8.41%       | Inspect non-numeric values; pandas dtype: str                                            |
| TOTAL_FLOOR                  | Total number of floors in the building              | numeric/text        | count                                  | 0.07%       | Potential prediction feature; pandas dtype: float64                                      |
| FEATURES                     | Property features or facilities                     | text                | Not applicable                         | 0.00%       | May contain multiple features in one field; pandas dtype: str                            |
| REGISTER_DATE                | Property registration date recorded in the dataset  | date/text           | Date                                   | 0.00%       | Verify date format and meaning; pandas dtype: str                                        |
| PROP_NAME                    | Property name                                       | text                | Not applicable                         | 14.66%      | Check overlap with other name fields; pandas dtype: str                                  |
| MIN_PRICE                    | Minimum price associated with a listing             | text/numeric        | Verify from source                     | 0.00%       | Check for overlap with PRICE; pandas dtype: int64                                        |
| MAX_PRICE                    | Maximum price associated with a listing             | text/numeric        | Verify from source                     | 0.00%       | Check for overlap with PRICE; pandas dtype: int64                                        |
| PRICE_SQFT                   | Price per square foot                               | text/numeric        | INR/sqft, subject to verification      | 0.00%       | Potentially derived from price and area; check for target leakage; pandas dtype: float64 |
| LISTING                      | Listing-related information or category             | text/categorical    | Not applicable                         | 0.00%       | Verify meaning from source documentation; pandas dtype: str                              |
| BUILDING_ID                  | Identifier associated with the building             | identifier          | Not applicable                         | 0.00%       | Not normally a direct prediction feature; pandas dtype: int64                            |
| CARPET_SQFT                  | Carpet area                                         | numeric/text        | square feet                            | 38.91%      | Check missing values and units; pandas dtype: float64                                    |
| SUPERBUILTUP_SQFT            | Super built-up area                                 | numeric/text        | square feet                            | 37.29%      | Check missing values and units; pandas dtype: float64                                    |
| BROKERAGE                    | Brokerage-related listing information               | text/numeric        | Verify from source                     | 0.76%       | Check units and meaning; pandas dtype: float64                                           |
| MAP_DETAILS                  | Map-related property information                    | text/structured     | Not applicable                         | 0.00%       | Inspect before extracting location features; pandas dtype: str                           |
| FSL_Data                     | Meaning not yet verified                            | str                 | Not verified                           | 0.00%       | Inspect sample values and source documentation; pandas dtype: str                        |
| MIN_AREA_SQFT                | Minimum listed area                                 | numeric/text        | square feet                            | 0.00%       | May be part of an area range; pandas dtype: float64                                      |
| MAX_AREA_SQFT                | Maximum listed area                                 | numeric/text        | square feet                            | 0.00%       | May be part of an area range; pandas dtype: float64                                      |
| FORMATTED                    | Meaning not yet verified                            | str                 | Not verified                           | 0.00%       | Inspect sample values and source documentation; pandas dtype: str                        |
| AMENITIES                    | Amenities associated with the property              | text                | Not applicable                         | 0.78%       | May be useful for feature engineering; pandas dtype: str                                 |
| TOP_USPS                     | Highlighted property selling points                 | text                | Not applicable                         | 23.59%      | May contain multiple values; pandas dtype: str                                           |
| PD_URL                       | Meaning not yet verified                            | str                 | Not verified                           | 0.02%       | Inspect sample values and source documentation; pandas dtype: str                        |
| EXPIRY_DATE                  | Listing expiry date                                 | date/text           | Date                                   | 0.00%       | Verify meaning and availability; pandas dtype: str                                       |
| GROUP_NAME                   | Group name associated with the listing              | text                | Not applicable                         | 0.00%       | Verify meaning from source documentation; pandas dtype: str                              |
| AREA                         | Listed property area                                | text/numeric        | Verify from source                     | 0.00%       | Inspect for ranges and inconsistent formats; pandas dtype: str                           |
| PRICE                        | Listed property price                               | text/numeric        | INR; verify units and transaction type | 0.00%       | Candidate target; raw values may contain Cr, lakh, or commas; pandas dtype: str          |
| PROP_HEADING                 | Meaning not yet verified                            | str                 | Not verified                           | 0.00%       | Inspect sample values and source documentation; pandas dtype: str                        |
| PROP_DETAILS_URL             | URL for property details                            | text/URL            | Not applicable                         | 0.00%       | Source listing reference; pandas dtype: str                                              |
| CLASS_HEADING                | Heading associated with the property classification | text                | Not applicable                         | 0.00%       | Verify meaning from source documentation; pandas dtype: str                              |
| CLASS_LABEL                  | Label associated with the property classification   | categorical/text    | Not applicable                         | 0.00%       | Verify category definitions; pandas dtype: str                                           |
| SECONDARY_TAGS               | Additional tags associated with the listing         | text/structured     | Not applicable                         | 0.00%       | May contain multiple categories; pandas dtype: str                                       |
| PROPERTY_IMAGES              | Property image information                          | text/structured     | Not applicable                         | 17.20%      | May contain multiple image references; pandas dtype: str                                 |
| THUMBNAIL_IMAGES             | Thumbnail image information                         | text/structured     | Not applicable                         | 17.20%      | May contain multiple image references; pandas dtype: str                                 |
| TOTAL_LANDMARK_COUNT         | Number of landmarks recorded for the property       | numeric             | count                                  | 0.62%       | Verify how landmarks were counted; pandas dtype: float64                                 |
| FORMATTED_LANDMARK_DETAILS   | Formatted information about nearby landmarks        | text                | Not applicable                         | 0.62%       | Inspect whether distances or landmark categories are available; pandas dtype: str        |
| CONTACT_NAME                 | Name of the listing contact                         | text                | Not applicable                         | 0.41%       | Personal/contact information; exclude from prediction features; pandas dtype: str        |
| CONTACT_COMPANY_NAME         | Company name associated with the contact            | text                | Not applicable                         | 0.05%       | Check relevance before using; pandas dtype: str                                          |
| DEALER_PHOTO_URL             | URL for the dealer's photo                          | text/URL            | Not applicable                         | 0.00%       | Not normally a prediction feature; pandas dtype: str                                     |
| SOCIETY_NAME                 | Name of the residential society                     | text                | Not applicable                         | 14.66%      | May contain missing values or high-cardinality categories; pandas dtype: str             |
| BUILDING_NAME                | Name of the building                                | text                | Not applicable                         | 14.66%      | May contain high-cardinality values; pandas dtype: str                                   |
| CITY_ID                      | Identifier associated with the city                 | identifier          | Not applicable                         | 0.00%       | May duplicate information in CITY; pandas dtype: int64                                   |
| LOCALITY_WO_CITY             | Locality name without the city name                 | categorical/text    | Not applicable                         | 0.00%       | Check overlap with LOCALITY; pandas dtype: str                                           |
| profile                      | Profile-related listing information                 | text/structured     | Not applicable                         | 0.00%       | Meaning should be verified; pandas dtype: str                                            |
| xid                          | External or source-specific identifier              | identifier          | Not applicable                         | 0.00%       | Not normally a direct prediction feature; pandas dtype: str                              |
| metadata                     | Additional metadata                                 | text/structured     | Not applicable                         | 0.00%       | Inspect contents before using; pandas dtype: str                                         |
| location                     | Location-related information                        | text/structured     | Not applicable                         | 0.00%       | Inspect format; may contain useful geographic information; pandas dtype: str             |
| BUILTUP_SQFT                 | Built-up area                                       | numeric/text        | square feet                            | 74.47%      | Check missing values and units; pandas dtype: float64                                    |
| SUPER_SQFT                   | Super area measurement                              | numeric/text        | square feet, subject to verification   | 91.83%      | Check whether this duplicates another area column; pandas dtype: float64                 |
| COMMON_FURNISHING_ATTRIBUTES | Common furnishing attributes                        | text/structured     | Not applicable                         | 100.00%     | May contain multiple values; pandas dtype: float64                                       |
| QUALITY_SCORE                | Quality score associated with the listing           | numeric             | Score; verify scale                    | 99.98%      | Check how the score was calculated; pandas dtype: float64                                |
| FURNISHING_ATTRIBUTES        | Detailed furnishing attributes                      | text/structured     | Not applicable                         | 99.98%      | May contain mixed data types; pandas dtype: str                                          |

## Data Quality Notes

- The original dataset is preserved in `data/raw/`.
- Missing percentages are calculated from the raw data before cleaning.
- Data types represent the current pandas representation of each column.
- Price units and transaction categories must be verified before
  preparing the prediction target.
- Listed prices should not automatically be interpreted as completed
  sale prices.
