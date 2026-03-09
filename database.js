// Replace these variables with your actual collected data
const name = /* user's name */
const mailId = /* user's email */
const date = /* booking date */
const phone = /* user's phone */

// Insert into Supabase
const { data, error } = await supabase
  .from('booking records')
  .insert([
    {
      Name: name,
      "Mail ID": mailId,
      Date: date,
      phone: phone
    }
  ]);

if (error) {
  // handle error (e.g., show a message)
} else {
  // handle success (e.g., show confirmation)
}
