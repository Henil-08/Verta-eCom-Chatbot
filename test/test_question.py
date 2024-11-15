testcases = [
    {
        "question": "What is the rating of the product?",
        "expected_answer": "The product has an average rating of 4.8 out of 5.",
        "follow_up_questions": "How does this rating compare to similar products?\nWhat factors contributed to this rating?\nAre there any specific reviews highlighting the rating?"
    },
    {
        "question": "How tall is it?",
        "expected_answer": "The product stands 3.75 inches tall.",
        "follow_up_questions": "Do you want to know if this is the standard size for FunKo figures?\nAre you interested in knowing about larger versions of this product?\nWould you like to explore how the size affects its display options?"
    },
    {
        "question": "Does it come in a box?",
        "expected_answer": "Yes, the product comes in a protective window display box.",
        "follow_up_questions": "Do you want to know if the box is suitable for gifting?\nWould you like to know about the branding on the packaging?\nAre you interested in knowing if the box can be reused for storage?"
    },
    {
        "question": "What do people say about it?",
        "expected_answer": "The product has an average rating of 4.8 out of 5 based on 1,234 reviews, with customers praising its design and collectible value.",
        "follow_up_questions": "Do you want to know about the most common positive reviews?\nWould you like to explore any complaints about the product?\nAre you interested in the recency of these reviews?"
    },
    {
        "question": "How much does it weigh?",
        "expected_answer": "The item weighs 4 ounces.",
        "follow_up_questions": "Do you want to know if it’s lightweight compared to similar collectibles?\nWould you like to check if the weight includes the packaging?\nAre you interested in how the weight affects shipping costs?"
    },
    {
        "question": "Is it good for gifting?",
        "expected_answer": "Yes, it’s an excellent gift for Lord of the Rings fans or FunKo POP! collectors.",
        "follow_up_questions": "Do you want to know if there’s a gift box option available?\nWould you like to explore what reviewers say about gifting this item?\nAre you interested in whether it can be personalized for gifting?"
    },
    {
        "question": "Is it durable?",
        "expected_answer": "Yes, the product is made of durable vinyl material and is frequently praised for its sturdiness.",
        "follow_up_questions": "Do you want to know if it can withstand accidental drops?\nWould you like to know if there are any complaints about wear and tear?\nAre you interested in whether it’s suitable for outdoor display?"
    },
    {
        "question": "What’s the price?",
        "expected_answer": "The product is priced at $10.99.",
        "follow_up_questions": "Do you want to know if there are any discounts available?\nWould you like to compare this price with similar items?\nAre you interested in whether the price includes shipping or tax?"
    },
    {
        "question": "Does it have limited availability?",
        "expected_answer": "No, this does not appear to be a limited edition.",
        "follow_up_questions": "Do you want to know about any limited-edition variants in the series?\nAre you interested in the store’s restocking frequency?\nWould you like to know what differentiates limited editions?"
    },
    {
        "question": "Can I put it on my desk?",
        "expected_answer": "Yes, the compact size and design make it perfect for desks, bookshelves, or display cases.",
        "follow_up_questions": "Do you want to know if it comes with a base for stability?\nWould you like to explore how it can be displayed in different positions?\nAre you interested in whether it is prone to tipping over?"
    },
    {
        "question": "Should I buy it?",
        "expected_answer": "This product is perfect for collectors and fans, offering a combination of detailed design, affordability, and collectible value.",
        "follow_up_questions": "Do you want to know what customer reviews highlight about its appeal?\nAre you interested in whether it’s considered a must-have for collectors?\nWould you like to explore if there are similar products with better value?"
    },
    {
        "question": "Is it waterproof?",
        "expected_answer": "No, the product is not waterproof.",
        "follow_up_questions": "Do you want to know if it’s resistant to minor spills?\nWould you like to know if moisture affects the paint quality?\nAre you interested in whether it can be displayed in humid environments?"
    },
    {
        "question": "What’s the weather like today?",
        "expected_answer": "I'm here to provide product-related information. For weather updates, please check a weather app or service.",
        "follow_up_questions": "Would you like to explore location-based services we offer?\nDo you want to know about weather-related products?\nAre you curious about other non-product queries I can assist with?"
    },
    {
        "question": "What is 2+2?",
        "expected_answer": "I'm here to provide product-related information. For general math queries, please use a calculator or search engine.",
        "follow_up_questions": "Do you want help calculating shipping costs?\nWould you like to know about price comparisons for similar items?\nAre you interested in budgeting for collectibles?"
    },
    {
        "question": "What materials are used in this product?",
        "expected_answer": "The product is made of durable vinyl material.",
        "follow_up_questions": "Is the vinyl recyclable?\nDoes the material affect durability?\nAre there any safety concerns with the material?"
    },
    {
        "question": "Can I use this as a paperweight?",
        "expected_answer": "Yes, its compact size and 4-ounce weight make it suitable as a paperweight, though it's primarily a collectible.",
        "follow_up_questions": "Do you want to know if it leaves marks on surfaces?\nWould you like to explore if the base is flat enough for stability?\nAre you interested in other multifunctional uses for this product?"
    },
    {
        "question": "Who is Frodo Baggins?",
        "expected_answer": "Frodo Baggins is the protagonist of the Lord of the Rings series, and this figure represents him in FunKo's collectible style.",
        "follow_up_questions": "Do you want to know about other Lord of the Rings characters in this series?\nWould you like to explore what makes this Frodo figure unique?\nAre you interested in Frodo’s popularity among collectors?"
    },
    {
        "question": "What’s included in the box?",
        "expected_answer": "The box includes the FunKo POP! Vinyl Figure of Frodo Baggins and a protective display box.",
        "follow_up_questions": "Do you want to know if there are any accessories included?\nWould you like to explore if the packaging includes additional branding?\nAre you interested in whether the display box is reusable?"
    },
    {
        "question": "Can it be used outdoors?",
        "expected_answer": "While durable, this collectible is best suited for indoor use to avoid exposure to weather elements.",
        "follow_up_questions": "Do you want to know if sunlight affects its paint quality?\nWould you like to explore protective coatings for outdoor use?\nAre you interested in the recommended conditions for display?"
    },
    {
        "question": "Can you help with my homework?",
        "expected_answer": "I'm here to assist with product-related questions. For homework help, consider using educational resources or tutors.",
        "follow_up_questions": "Do you want information about Frodo Baggins’ cultural significance?\nWould you like to know about related Lord of the Rings books or materials?\nAre you curious about other educational products we offer?"
    },
    {
        "question": "Is it edible?",
        "expected_answer": "No, this is a collectible and not intended for consumption.",
        "follow_up_questions": "Do you want to know if there are food-themed FunKo collectibles?\nWould you like to explore edible gifts in the Lord of the Rings merchandise?\nAre you interested in thematic party supplies?"
    },
    {
    "question": "Can kids use it?",
        "expected_answer": "Yes, this FunKo POP! figure is suitable for kids over the age of 3. However, as it is primarily a collectible, it may not be as durable as a standard toy.",
        "follow_up_questions": "Do you want to know if it has any small parts that could be a choking hazard?\nWould you like recommendations for kid-friendly FunKo collectibles?\nAre you interested in similar collectibles specifically designed for children?"
    },
    {
        "question": "Does it come in different colors?",
        "expected_answer": "No, this specific Frodo Baggins FunKo POP! figure only comes in the standard color scheme based on the character’s appearance in the Lord of the Rings movies.",
        "follow_up_questions": "Do you want to know about other characters in the series with varied colors?\nWould you like to explore limited edition versions with alternate color schemes?\nAre you interested in FunKo POP! figures with custom paint options?"
    },
    {
        "question": "Does it come in different materials?",
        "expected_answer": "No, the FunKo POP! Frodo Baggins figure is exclusively made of durable vinyl material. There are no versions available in different materials.",
        "follow_up_questions": "Are you interested in other collectibles made from different materials?\nWould you like to know about the benefits of vinyl for collectibles?\nDo you want information on how to care for vinyl figures?"
    },
    {
        "question": "Does the color fade?",
        "expected_answer": "The FunKo POP! Frodo Baggins figure is designed to have long-lasting color that does not fade easily under normal display conditions. However, exposure to direct sunlight for prolonged periods may affect the color over time.",
        "follow_up_questions": "Would you like tips on how to preserve the color?\nDo you want to know about the materials used in the figure's paint?\nAre you interested in knowing how long the color typically lasts?"
    },
    {
        "question": "What's your name?",
        "expected_answer": "I'm Verta, your intelligent eCommerce companion here to assist you with any questions related to this product and similar items.",
        "follow_up_questions": "Can I help you with anything specific about this product?\nWould you like to know more about our services?\nIs there anything else you'd like assistance with?"
    }
]
